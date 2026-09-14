"""CAD signal-feature modelling and prospective validation.

The script deliberately keeps clinical labels and model outputs separate: labels are
only derived from questionnaire/clinical-diagnosis text, never from device results.
It produces the requested ten physiological features and fits an XGBoost model with
max_depth selected only from 3, 4, 5, and 6 by grouped cross-validation.
"""
from __future__ import annotations

import argparse, json, re, sys, time, zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

FEATURES = ["resting_hr_bpm", "sdnn_ms", "rmssd_ms", "lf_hf_ratio", "perfusion_index",
            "augmentation_index", "primary_reflected_peak_delay_ms", "st_deviation_mv",
            "qtc_ms", "pulse_transit_time_ms"]

def log(message: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

def rows(folder: Path):
    for f in sorted(folder.glob("records_*.jsonl")):
        with f.open(encoding="utf-8") as h:
            for line in h:
                try: yield json.loads(line)
                except json.JSONDecodeError: continue

def safe_signal_from_zip(path: Path, kind: str):
    """Read legacy ECG/PPG NDJSON attachment. Returns signal and estimated Hz."""
    try:
        with zipfile.ZipFile(path) as z:
            names = [x for x in z.namelist() if x != "info.json"]
            raw = z.read(names[0]).decode("utf-8", errors="ignore").splitlines()
    except (OSError, KeyError, zipfile.BadZipFile): return None, None
    vals, times = [], []
    for line in raw:
        try:
            d = json.loads(line)
            frame = d.get("timeFrame", {})
            x = d.get("greenLight") if kind == "ppg" else d.get("ecg", d.get("voltage"))
            unit = ""
            # ECG exports store a complete 30-second trace as one JSON object:
            # voltage=[{unit: "μV", value: ...}, ...].  It is not NDJSON.
            if kind == "ecg" and isinstance(x, list) and x and isinstance(x[0], dict):
                vals.extend(float(item["value"]) / 1000.0 for item in x if item.get("value") is not None)
                start = float(frame.get("timestamp", np.nan))
                # The device ECG trace has 15,000 samples per 30 s (500 Hz).
                times.extend(start + np.arange(len(x)) * 2.0)
                continue
            if isinstance(x, list): x = x[0] if x else None
            if isinstance(x, dict):
                unit, x = x.get("unit", ""), x.get("value")
            if x is None: x = d.get("value")
            if x is not None:
                # Native ECG attachments store voltage in microvolts; all ECG-derived
                # features use millivolts to make ST deviation clinically interpretable.
                value = float(x) / 1000.0 if kind == "ecg" and ("V" in unit or abs(float(x)) > 20) else float(x)
                vals.append(value); times.append(float(frame.get("timestamp", np.nan)))
        except (ValueError, TypeError, json.JSONDecodeError): pass
    if len(vals) < 300: return None, None
    dt = np.nanmedian(np.diff(times))
    fs = 1000.0 / dt if np.isfinite(dt) and 1 < dt < 1000 else (100.0 if kind == "ppg" else 500.0)
    return np.asarray(vals), fs

def bandpass(x, fs, lo, hi):
    if len(x) < 30: return x
    b, a = butter(3, [lo/(fs/2), min(hi/(fs/2), .99)], btype="band")
    return filtfilt(b, a, x)

def ppg_features(ppg, fs):
    x = bandpass(np.asarray(ppg), fs, .5, 8)
    prom = max(np.std(x) * .35, 1e-9)
    peaks, _ = find_peaks(x, distance=int(fs*.35), prominence=prom)
    if len(peaks) < 4: return {k: np.nan for k in FEATURES[:7]}, np.array([]), peaks
    rr = np.diff(peaks) / fs
    rr = rr[(rr > .35) & (rr < 2.0)]
    hr = 60 / np.median(rr) if len(rr) else np.nan
    sdnn = np.std(rr, ddof=1)*1000 if len(rr)>1 else np.nan
    rmssd = np.sqrt(np.mean(np.diff(rr)**2))*1000 if len(rr)>2 else np.nan
    f, psd = welch(1/rr if len(rr)>3 else np.array([0]), fs=4, nperseg=min(64, max(4,len(rr))))
    power = lambda lo, hi: np.trapz(psd[(f>=lo)&(f<hi)], f[(f>=lo)&(f<hi)])
    hf, lf = power(.15,.4), power(.04,.15)
    pi = (np.percentile(x,95)-np.percentile(x,5))/(abs(np.mean(x))+1e-9)*100
    delays=[]; ais=[]
    for a,b in zip(peaks[:-1], peaks[1:]):
        seg=x[a:b]; trough=a+np.argmin(x[a:b])
        sub,_=find_peaks(x[a+int(.08*fs):trough], prominence=prom*.1)
        if len(sub):
            r=a+int(.08*fs)+sub[-1]; delays.append((r-a)/fs*1000)
            ais.append((x[r]-x[a])/(x[a]-x[trough]+1e-9)*100)
    out=dict(resting_hr_bpm=hr, sdnn_ms=sdnn, rmssd_ms=rmssd,
             lf_hf_ratio=lf/(hf+1e-9), perfusion_index=pi,
             augmentation_index=np.nanmedian(ais) if ais else np.nan,
             primary_reflected_peak_delay_ms=np.nanmedian(delays) if delays else np.nan)
    return out, rr, peaks

def ecg_features(ecg, fs):
    if ecg is None: return {"st_deviation_mv":np.nan,"qtc_ms":np.nan}, np.array([])
    x=bandpass(ecg,fs,.5,40); peaks,_=find_peaks(x, distance=int(.35*fs), prominence=max(np.std(x)*.7,1e-8))
    if len(peaks)<3: return {"st_deviation_mv":np.nan,"qtc_ms":np.nan}, peaks
    rr=np.diff(peaks)/fs; qtc=[]; st=[]
    for i,p in enumerate(peaks[:-1]):
        end=peaks[i+1]; base=np.median(x[max(0,p-int(.20*fs)):max(1,p-int(.08*fs))])
        st.append(np.median(x[min(end-1,p+int(.08*fs)):min(end,p+int(.12*fs))])-base)
        # Conservative T-end proxy: last low-slope point before next QRS.
        t=x[p+int(.12*fs):end-int(.08*fs)]
        if len(t)>5:
            qtc.append((np.argmax(np.abs(t-base))/fs+.12)/np.sqrt(max(rr[i],.35))*1000)
    return {"st_deviation_mv":np.nanmedian(st),"qtc_ms":np.nanmedian(qtc) if qtc else np.nan},peaks

def feature_row(ppg, pfs, ecg=None, efs=None, ppg_start_ms=None, ecg_start_ms=None):
    out,_,pp=ppg_features(ppg,pfs); ef,ep=ecg_features(ecg,efs) if ecg is not None else ({"st_deviation_mv":np.nan,"qtc_ms":np.nan},np.array([]))
    out.update(ef)
    # PTT uses absolute time, not sample indices: ECG and PPG are acquired at
    # 500 Hz and 100 Hz respectively.  Their record timestamps establish alignment.
    if len(ep) and len(pp) and ppg_start_ms is not None and ecg_start_ms is not None:
        ppg_peak_time = pp / pfs + (float(ppg_start_ms) - float(ecg_start_ms)) / 1000.0
        lag=[]
        for q_time in ep / efs:
            cand=ppg_peak_time[ppg_peak_time>=q_time]
            if len(cand) and cand[0]-q_time < .5: lag.append((cand[0]-q_time)*1000)
        out["pulse_transit_time_ms"]=np.nanmedian(lag) if lag else np.nan
    else: out["pulse_transit_time_ms"]=np.nan
    return out

def is_cad(text):
    t=str(text).lower()
    return int(any(k in t for k in ["冠心病","冠状动脉粥样硬化","心绞痛","心肌梗死","pci","cabg","coronary","cad"]))

def main(a):
    train=Path(a.train); attach=Path(a.train_attachments); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    log("Stage 1/5: reading questionnaire labels")
    q=list(rows(train/"t_ae32d126_vascularquestionnairedata_nzqftz5h"))
    labels={r["healthid"]:is_cad(r.get("isDiagnosedCardiovascularDiseases","")) for r in q if r.get("healthid")}
    log(f"Labels loaded: {len(labels):,} participants; CAD positive: {sum(labels.values()):,}")
    # one most recent PPG and ECG record per labelled participant, matched by group.
    ppg={}; ecg={}
    log("Stage 2/5: indexing PPG records")
    for n,r in enumerate(rows(train/"t_ae32d126_vascularppg_nh8sf1zv"), 1):
        h=r.get("healthid");
        if h in labels and (h not in ppg or r.get("recordtime",0)>ppg[h].get("recordtime",0)): ppg[h]=r
        if n % 50000 == 0: log(f"PPG indexed: {n:,} records scanned; {len(ppg):,} labelled participants matched")
    log(f"PPG index complete: {len(ppg):,} labelled participants")
    log("Stage 3/5: indexing ECG records")
    for n,r in enumerate(rows(train/"t_ae32d126_vascularecg_ccmfayhr"), 1):
        h=r.get("healthid");
        if h in labels and (h not in ecg or r.get("recordtime",0)>ecg[h].get("recordtime",0)): ecg[h]=r
        if n % 50000 == 0: log(f"ECG indexed: {n:,} records scanned; {len(ecg):,} labelled participants matched")
    log(f"ECG index complete: {len(ecg):,} labelled participants")
    log("Stage 4/5: extracting patient-level signal features")
    dataset=[]; missing_ppg=0; unreadable_ppg=0; matched_ecg=0; unreadable_ecg=0
    for n,(h,r) in enumerate(ppg.items(), 1):
        z=attach/"ppg"/f"ppg_{pd.to_datetime(r['recordtime'],unit='ms').strftime('%Y%m%d')}"/Path(r['ppg']).name
        if not z.is_file():
            missing_ppg += 1
            continue
        sig,fs=safe_signal_from_zip(z,"ppg")
        if sig is None:
            unreadable_ppg += 1
            continue
        er=ecg.get(h); es=efs=None
        if er and er.get("groupid")==r.get("groupid"):
            ez=attach/"ecg"/f"ecg_{pd.to_datetime(er['recordtime'],unit='ms').strftime('%Y%m%d')}"/Path(er['ecg']).name
            if ez.is_file():
                es,efs=safe_signal_from_zip(ez,"ecg")
                matched_ecg += int(es is not None)
                unreadable_ecg += int(es is None)
        d=feature_row(sig,fs,es,efs,r.get("recordtime"),er.get("recordtime") if er else None); d.update(healthid=h,label=labels[h]); dataset.append(d)
        if n % 100 == 0 or n == len(ppg): log(f"Features: {n:,}/{len(ppg):,}; retained {len(dataset):,}; ECG paired {matched_ecg:,}")
    df=pd.DataFrame(dataset, columns=FEATURES+["healthid","label"])
    df.to_csv(out/"training_features.csv",index=False,encoding="utf-8-sig")
    audit = {"questionnaire_labels":len(labels), "label_positive":sum(labels.values()),
             "ppg_candidates":len(ppg), "ecg_candidates":len(ecg),
             "missing_ppg_attachment":missing_ppg, "unreadable_ppg":unreadable_ppg,
             "matched_ecg":matched_ecg, "unreadable_ecg":unreadable_ecg,
             "feature_rows":len(df)}
    (out/"cohort_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    classes=df["label"].value_counts().to_dict()
    if df.empty or len(classes)<2:
        raise RuntimeError(f"No trainable two-class cohort. Audit written to {out/'cohort_audit.json'}: {audit}; classes={classes}")
    X=SimpleImputer(strategy="median",add_indicator=True).fit_transform(df[FEATURES]); y=df.label.to_numpy(); groups=df.healthid.to_numpy()
    log(f"Stage 5/5: grouped XGBoost cross-validation on {len(df):,} samples; classes={classes}")
    scores={}
    for depth in range(3,7):
        fold=[]
        for tr,va in GroupKFold(n_splits=min(5,len(np.unique(groups)))).split(X,y,groups):
            if len(np.unique(y[tr]))<2: continue
            m=XGBClassifier(max_depth=depth,n_estimators=250,learning_rate=.04,subsample=.8,colsample_bytree=.8,eval_metric="logloss",random_state=42)
            m.fit(X[tr],y[tr]); fold.append(roc_auc_score(y[va],m.predict_proba(X[va])[:,1]) if len(np.unique(y[va]))==2 else np.nan)
        scores[depth]=np.nanmean(fold)
        log(f"max_depth={depth}: grouped CV AUC={scores[depth]:.4f}")
    depth=max(scores,key=scores.get); model=XGBClassifier(max_depth=depth,n_estimators=300,learning_rate=.04,subsample=.8,colsample_bytree=.8,eval_metric="logloss",random_state=42).fit(X,y)
    model.save_model(out/"cad_xgboost.json")
    pd.DataFrame({"max_depth":list(scores),"grouped_cv_auc":list(scores.values())}).to_csv(out/"depth_selection.csv",index=False)
    (out/"README.txt").write_text(f"Training cohort={len(df)}; positive={int(y.sum())}; selected max_depth={depth}.\nValidation requires matching E attachments to clinical IDs; no predictions are emitted without an explicit patient-to-zip mapping.\n",encoding="utf-8")
    log(f"Completed. Selected max_depth={depth}; results written to {out}")

if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument('--train',default=r'X:\WangYing\table_data\data'); p.add_argument('--train-attachments',default=r'X:\WangYing\huawei_attachments'); p.add_argument('--output',default=r'E:\data\cad_model_output'); args=p.parse_args()
 if not Path(args.train).is_dir(): raise FileNotFoundError(f"Training directory does not exist: {args.train}")
 if not Path(args.train_attachments).is_dir(): raise FileNotFoundError(f"Attachment directory does not exist: {args.train_attachments}")
 main(args)
