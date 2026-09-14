"""Fast 5-fold CV baseline - processes first 500 zips for quick results."""
from __future__ import annotations

import csv, json, os, time, zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

FEATURES = ["resting_hr_bpm", "sdnn_ms", "rmssd_ms", "lf_hf_ratio", "perfusion_index",
            "augmentation_index", "primary_reflected_peak_delay_ms", "st_deviation_mv",
            "qtc_ms", "pulse_transit_time_ms"]

ATTACH_DIR = Path(r"E:\data\dataset\attachments\attachments")
MAPPING_FILE = Path(r"E:\data\dataset\attachments\标签对应表.txt")
OUTPUT_DIR = Path(r"E:\data\attachments_cv_output")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def is_cad(text):
    t = str(text).lower()
    return int(any(k in t for k in ["冠心病","冠状动脉粥样硬化","心绞痛","心肌梗死","pci","cabg","coronary","cad"]))

def bandpass(x, fs, lo, hi):
    if len(x) < 30: return x
    b, a = butter(3, [lo/(fs/2), min(hi/(fs/2),.99)], btype="band")
    return filtfilt(b, a, x)

def ppg_features(ppg, fs):
    x = bandpass(np.asarray(ppg, dtype=float), fs, 0.5, 8)
    prom = max(np.std(x)*0.35, 1e-9)
    peaks, _ = find_peaks(x, distance=int(fs*0.35), prominence=prom)
    if len(peaks) < 4:
        return {k: np.nan for k in FEATURES}
    rr = np.diff(peaks)/fs
    rr = rr[(rr > 0.35) & (rr < 2.0)]
    hr = 60/np.median(rr) if len(rr) else np.nan
    sdnn = np.std(rr, ddof=1)*1000 if len(rr) > 1 else np.nan
    rmssd = np.sqrt(np.mean(np.diff(rr)**2))*1000 if len(rr) > 2 else np.nan
    f, psd = welch(1/rr if len(rr) > 3 else np.array([0]), fs=4, nperseg=min(64, max(4, len(rr))))
    def power(lo_, hi_):
        m = (f >= lo_) & (f < hi_)
        return np.trapezoid(psd[m], f[m]) if m.any() else 0.0
    hf, lf = power(0.15, 0.4), power(0.04, 0.15)
    pi = (np.percentile(x,95)-np.percentile(x,5))/(abs(np.mean(x))+1e-9)*100
    delays, ais = [], []
    for a, b in zip(peaks[:-1], peaks[1:]):
        trough = a + np.argmin(x[a:b])
        sub, _ = find_peaks(x[a+int(0.08*fs):trough], prominence=prom*0.1)
        if len(sub):
            r = a + int(0.08*fs) + sub[-1]
            delays.append((r-a)/fs*1000)
            ais.append((x[r]-x[a])/(x[a]-x[trough]+1e-9)*100)

    st_devs = []
    for i in range(len(peaks) - 1):
        st_start = peaks[i] + int(fs * 0.08)
        st_end = peaks[i] + int(fs * 0.12)
        if st_end < len(x):
            st_devs.append(np.mean(x[st_start:st_end]) - np.mean(x))
    st_deviation = np.nanmedian(st_devs) if st_devs else np.nan

    if len(rr) > 1:
        mean_rr = np.mean(rr)
        qt_interval = mean_rr * 0.4
        qtc = qt_interval / np.sqrt(mean_rr) * 1000 if mean_rr > 0 else np.nan
    else:
        qtc = np.nan

    pulse_transit_time = np.mean(np.diff(peaks) / fs) * 1000 if len(peaks) >= 2 else np.nan

    return dict(resting_hr_bpm=hr, sdnn_ms=sdnn, rmssd_ms=rmssd,
                lf_hf_ratio=lf/(hf+1e-9), perfusion_index=pi,
                augmentation_index=np.nanmedian(ais) if ais else np.nan,
                primary_reflected_peak_delay_ms=np.nanmedian(delays) if delays else np.nan,
                st_deviation_mv=st_deviation, qtc_ms=qtc,
                pulse_transit_time_ms=pulse_transit_time)

def read_zip_sensor(zip_path):
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = [x for x in z.namelist() if x != "info.json"]
            if not names: return None, None
            raw = z.read(names[0]).decode("utf-8", errors="ignore").splitlines()
    except Exception:
        return None, None
    # Tab-separated format
    header = raw[0].split('\t')
    ppg_time_idx = header.index('PPG_TIME') if 'PPG_TIME' in header else -1
    ppg9_idx = header.index('PPG9') if 'PPG9' in header else -1
    if ppg_time_idx >= 0 and ppg9_idx >= 0:
        ts_val_pairs = []
        for line in raw[1:]:
            parts = line.split('\t')
            if ppg_time_idx >= len(parts) or ppg9_idx >= len(parts): continue
            try:
                t = int(parts[ppg_time_idx]); v = float(parts[ppg9_idx])
                if t > 0 and v != 0: ts_val_pairs.append((t, v))
            except: pass
        if len(ts_val_pairs) >= 300:
            vals = np.array([p[1] for p in ts_val_pairs], dtype=float)
            ts_arr = np.array([p[0] for p in ts_val_pairs], dtype=float)
            unique_ts = np.unique(ts_arr)
            if len(unique_ts) >= 2:
                median_block_gap = np.median(np.diff(unique_ts))
                spb = len(ts_val_pairs) / len(unique_ts)
                fs = spb*1000.0/median_block_gap if median_block_gap > 0 else 100.0
                return vals, max(min(fs, 500), 10)
    # JSON fallback
    vals, times = [], []
    for line in raw:
        try:
            d = json.loads(line)
            frame = d.get("timeFrame", {})
            x = d.get("greenLight")
            if isinstance(x, list): x = x[0] if x else None
            if isinstance(x, dict): x = x.get("value")
            if x is None: x = d.get("value")
            if x is not None:
                vals.append(float(x)); times.append(float(frame.get("timestamp", np.nan)))
        except: pass
    if len(vals) < 300: return None, None
    dt = np.nanmedian(np.diff(times))
    fs = 1000.0/dt if np.isfinite(dt) and 1 < dt < 1000 else 100.0
    return np.asarray(vals), fs

def load_mapping():
    ts_to_ext = {}
    with open(MAPPING_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f); next(reader)
        for row in reader:
            data_id = row[0]; external_id = row[3] if len(row) > 3 else ""
            parts = data_id.split('_')
            if len(parts) >= 2: ts_to_ext[parts[1]] = external_id
    return ts_to_ext

def extract_one(args):
    zip_path, ts_str, ts_to_ext = args
    external_id = ts_to_ext.get(ts_str, "")
    if not external_id: return None
    sig, fs = read_zip_sensor(zip_path)
    if sig is None or fs is None: return None
    feats = ppg_features(sig, fs)
    feats["st_deviation_mv"] = np.nan; feats["qtc_ms"] = np.nan; feats["pulse_transit_time_ms"] = np.nan
    return {**{f: feats.get(f, np.nan) for f in FEATURES}, "external_id": external_id, "label": is_cad(external_id)}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = OUTPUT_DIR / "extracted_features.csv"

    if cache_file.exists():
        log(f"Loading cached features from {cache_file}")
        df = pd.read_csv(cache_file)
        log(f"Loaded {len(df)} samples")
    else:
        log("Loading mapping table...")
        ts_to_ext = load_mapping()
        log(f"Loaded {len(ts_to_ext)} mappings")

        zip_files = sorted(f for f in os.listdir(ATTACH_DIR) if f.endswith('.zip'))
        log(f"Found {len(zip_files)} zip files")

        # Build task list
        tasks = []
        for zn in zip_files:
            zp = ATTACH_DIR / zn
            try:
                with zipfile.ZipFile(zp) as z:
                    names = [x for x in z.namelist() if x != "info.json"]
                    if not names: continue
                    fname = names[0].replace('.txt', '')
                    parts = fname.split('_')
                    if len(parts) >= 3:
                        tasks.append((zp, parts[2], ts_to_ext))
            except: pass

        log(f"Will process {len(tasks)} zips with valid timestamps")
        t0 = time.time()

        # Process sequentially for reliability
        results = []
        for i, (zp, ts, t2e) in enumerate(tasks):
            ext = t2e.get(ts, "")
            if not ext: continue
            sig, fs = read_zip_sensor(zp)
            if sig is None or fs is None: continue
            feats = ppg_features(sig, fs)
            results.append({**{f: feats.get(f, np.nan) for f in FEATURES}, "external_id": ext, "label": is_cad(ext)})
            if (i+1) % 200 == 0:
                elapsed = time.time()-t0
                log(f"  {i+1}/{len(tasks)} done ({len(results)} matched); {elapsed:.0f}s elapsed")

        log(f"Extracted {len(results)} samples in {time.time()-t0:.0f}s")
        df = pd.DataFrame(results)
        df.to_csv(cache_file, index=False, encoding="utf-8-sig")
        log(f"Saved to {cache_file}")

    # CV
    label_counts = df["label"].value_counts()
    log(f"Labels: {dict(label_counts)}")
    if len(label_counts) < 2:
        log("ERROR: Need 2+ classes"); return

    X = SimpleImputer(strategy="median", add_indicator=True).fit_transform(df[FEATURES])
    y = df["label"].values
    log(f"Feature matrix: {X.shape}")

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_scores, all_preds, all_labels = [], [], []

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            log(f"  Fold {fold+1}: skipped"); continue
        m = XGBClassifier(max_depth=4, n_estimators=300, learning_rate=0.04,
                          subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                          random_state=42, n_jobs=-1)
        m.fit(X[tr], y[tr])
        proba = m.predict_proba(X[va])[:,1]
        preds = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y[va], proba)
        all_scores.append(auc); all_preds.extend(preds); all_labels.extend(y[va])
        log(f"  Fold {fold+1}: AUC={auc:.4f}, n={len(y[va])}, pos={int(y[va].sum())}")

    mean_auc = np.mean(all_scores); std_auc = np.std(all_scores)
    log(f"\n{'='*60}")
    log(f"5-Fold CV Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
    log(f"Folds: {[f'{s:.4f}' for s in all_scores]}")
    log(f"N={len(y)}, pos={int(y.sum())}, neg={int(len(y)-y.sum())}")
    if all_preds:
        log(f"\n{classification_report(all_labels, all_preds, target_names=['Neg','Pos'])}")

    (OUTPUT_DIR/"cv_results.json").write_text(json.dumps({
        "mean_auc": float(mean_auc), "std_auc": float(std_auc),
        "fold_aucs": [float(s) for s in all_scores],
        "n": len(y), "pos": int(y.sum()), "neg": int(len(y)-y.sum())
    }, indent=2), encoding="utf-8")
    log(f"Saved results to {OUTPUT_DIR/'cv_results.json'}")

if __name__ == "__main__":
    main()
