"""5-fold cross-validation baseline using only attachments data.

Reads zip files from E:\data\dataset\attachments\attachments\,
maps them to labels via the mapping table, extracts PPG features,
and runs stratified 5-fold CV with XGBoost.
"""
from __future__ import annotations

import csv, json, os, sys, time, zipfile
from pathlib import Path

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
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def is_cad(text):
    t = str(text).lower()
    return int(any(k in t for k in ["冠心病", "冠状动脉粥样硬化", "心绞痛", "心肌梗死", "pci", "cabg", "coronary", "cad"]))


def bandpass(x, fs, lo, hi):
    if len(x) < 30:
        return x
    b, a = butter(3, [lo / (fs / 2), min(hi / (fs / 2), .99)], btype="band")
    return filtfilt(b, a, x)


def ppg_features(ppg, fs):
    x = bandpass(np.asarray(ppg, dtype=float), fs, 0.5, 8)
    prom = max(np.std(x) * 0.35, 1e-9)
    peaks, _ = find_peaks(x, distance=int(fs * 0.35), prominence=prom)
    if len(peaks) < 4:
        return {k: np.nan for k in FEATURES}
    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.35) & (rr < 2.0)]
    hr = 60 / np.median(rr) if len(rr) else np.nan
    sdnn = np.std(rr, ddof=1) * 1000 if len(rr) > 1 else np.nan
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000 if len(rr) > 2 else np.nan
    f, psd = welch(1 / rr if len(rr) > 3 else np.array([0]), fs=4,
                   nperseg=min(64, max(4, len(rr))))
    def power(lo_, hi_):
        mask = (f >= lo_) & (f < hi_)
        return np.trapz(psd[mask], f[mask]) if mask.any() else 0.0
    hf, lf = power(0.15, 0.4), power(0.04, 0.15)
    pi = (np.percentile(x, 95) - np.percentile(x, 5)) / (abs(np.mean(x)) + 1e-9) * 100
    delays, ais = [], []
    for a, b in zip(peaks[:-1], peaks[1:]):
        trough = a + np.argmin(x[a:b])
        sub, _ = find_peaks(x[a + int(0.08 * fs):trough], prominence=prom * 0.1)
        if len(sub):
            r = a + int(0.08 * fs) + sub[-1]
            delays.append((r - a) / fs * 1000)
            ais.append((x[r] - x[a]) / (x[a] - x[trough] + 1e-9) * 100)

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
                lf_hf_ratio=lf / (hf + 1e-9), perfusion_index=pi,
                augmentation_index=np.nanmedian(ais) if ais else np.nan,
                primary_reflected_peak_delay_ms=np.nanmedian(delays) if delays else np.nan,
                st_deviation_mv=st_deviation, qtc_ms=qtc,
                pulse_transit_time_ms=pulse_transit_time)


def read_zip_sensor(zip_path):
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = [x for x in z.namelist() if x != "info.json"]
            if not names:
                return None, None
            raw = z.read(names[0]).decode("utf-8", errors="ignore").splitlines()
    except Exception:
        return None, None

    # Try tab-separated format first (newer zips)
    header = raw[0].split('\t')
    ppg_time_idx = header.index('PPG_TIME') if 'PPG_TIME' in header else -1
    ppg9_idx = header.index('PPG9') if 'PPG9' in header else -1

    if ppg_time_idx >= 0 and ppg9_idx >= 0:
        ts_val_pairs = []
        for line in raw[1:]:
            parts = line.split('\t')
            if ppg_time_idx >= len(parts) or ppg9_idx >= len(parts):
                continue
            try:
                t = int(parts[ppg_time_idx])
                v = float(parts[ppg9_idx])
                if t > 0 and v != 0:
                    ts_val_pairs.append((t, v))
            except (ValueError, IndexError):
                continue
        if len(ts_val_pairs) >= 300:
            vals = np.array([p[1] for p in ts_val_pairs], dtype=float)
            ts_arr = np.array([p[0] for p in ts_val_pairs], dtype=float)
            unique_ts = np.unique(ts_arr)
            if len(unique_ts) >= 2:
                median_block_gap = np.median(np.diff(unique_ts))
                samples_per_block = len(ts_val_pairs) / len(unique_ts)
                fs = samples_per_block * 1000.0 / median_block_gap if median_block_gap > 0 else 100.0
                fs = max(min(fs, 500), 10)
                return vals, fs

    # Fallback: JSON NDJSON format (legacy zips)
    vals, times = [], []
    for line in raw:
        try:
            d = json.loads(line)
            frame = d.get("timeFrame", {})
            x = d.get("greenLight")
            if isinstance(x, list):
                x = x[0] if x else None
            if isinstance(x, dict):
                x = x.get("value")
            if x is None:
                x = d.get("value")
            if x is not None:
                vals.append(float(x))
                times.append(float(frame.get("timestamp", np.nan)))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    if len(vals) < 300:
        return None, None
    dt = np.nanmedian(np.diff(times))
    fs = 1000.0 / dt if np.isfinite(dt) and 1 < dt < 1000 else 100.0
    return np.asarray(vals), fs


def load_mapping():
    ts_to_ext = {}
    with open(MAPPING_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data_id = row[0]
            external_id = row[3] if len(row) > 3 else ""
            parts = data_id.split('_')
            if len(parts) >= 2:
                ts_to_ext[parts[1]] = external_id
    return ts_to_ext


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for cached features
    cache_file = OUTPUT_DIR / "extracted_features.csv"
    if cache_file.exists():
        log(f"Found cached features at {cache_file}, loading...")
        df = pd.read_csv(cache_file)
        log(f"Loaded {len(df)} cached samples")
    else:
        log("Loading mapping table...")
        ts_to_ext = load_mapping()
        log(f"Loaded {len(ts_to_ext)} timestamp->external_id mappings")

        zip_files = sorted(f for f in os.listdir(ATTACH_DIR) if f.endswith('.zip'))
        log(f"Found {len(zip_files)} zip files")

        # Extract features from all zips
        results = []
        processed = 0
        errors = 0
        matched = 0
        unmatched = 0
        t0 = time.time()

        for i, zip_name in enumerate(zip_files):
            zip_path = ATTACH_DIR / zip_name
            try:
                # Get timestamp from inside zip filename
                with zipfile.ZipFile(zip_path) as z:
                    names = [x for x in z.namelist() if x != "info.json"]
                    if not names:
                        errors += 1
                        continue
                    fname = names[0].replace('.txt', '')
                    parts = fname.split('_')
                    if len(parts) < 3:
                        errors += 1
                        continue
                    ts_str = parts[2]

                external_id = ts_to_ext.get(ts_str, "")
                if not external_id:
                    unmatched += 1
                    continue

                sig, fs = read_zip_sensor(zip_path)
                if sig is None or fs is None:
                    errors += 1
                    continue

                feats = ppg_features(sig, fs)

                label = is_cad(external_id)

                row = {f: feats.get(f, np.nan) for f in FEATURES}
                row["external_id"] = external_id
                row["label"] = label
                row["zip_file"] = zip_name
                results.append(row)
                matched += 1
                processed += 1

            except Exception:
                errors += 1
                continue

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (i+1) / elapsed
                eta = (len(zip_files) - i - 1) / rate
                log(f"Progress: {i+1}/{len(zip_files)} zips; {matched} matched; {errors} errors; rate={rate:.1f}/s; ETA={eta:.0f}s")

        log(f"Feature extraction: {processed} samples extracted, {errors} errors, {unmatched} unmatched timestamps")

        if not results:
            log("ERROR: No results to process!")
            return

        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_DIR / "extracted_features.csv", index=False, encoding="utf-8-sig")
        log(f"Saved features to {OUTPUT_DIR / 'extracted_features.csv'}")

    # Label distribution
    label_counts = df["label"].value_counts()
    log(f"Label distribution: {dict(label_counts)}")

    if len(label_counts) < 2:
        log("ERROR: Need at least 2 classes for cross-validation!")
        return

    # Prepare features
    X = SimpleImputer(strategy="median", add_indicator=True).fit_transform(df[FEATURES])
    y = df["label"].values

    log(f"Feature matrix shape: {X.shape}")

    # 5-fold stratified cross-validation
    log("Running 5-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_scores = []
    all_preds = []
    all_labels = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            log(f"  Fold {fold_idx+1}: skipped (insufficient class diversity)")
            continue

        model = XGBClassifier(
            max_depth=4, n_estimators=300, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_val)[:, 1]
        preds = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_val, proba)
        all_scores.append(auc)
        all_preds.extend(preds)
        all_labels.extend(y_val)

        log(f"  Fold {fold_idx+1}: AUC={auc:.4f}, n_val={len(y_val)}, pos={int(y_val.sum())}")

    mean_auc = np.mean(all_scores)
    std_auc = np.std(all_scores)
    log(f"\n{'='*60}")
    log(f"5-Fold CV Results:")
    log(f"  Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
    log(f"  Individual fold AUCs: {[f'{s:.4f}' for s in all_scores]}")
    log(f"  Total samples: {len(y)}")
    log(f"  Positive samples: {int(y.sum())}")
    log(f"  Negative samples: {int(len(y) - y.sum())}")

    if all_preds:
        report = classification_report(all_labels, all_preds, target_names=["Negative", "Positive"])
        log(f"\nClassification Report:\n{report}")

    log(f"{'='*60}")

    # Save results
    results_summary = {
        "mean_auc": float(mean_auc),
        "std_auc": float(std_auc),
        "fold_aucs": [float(s) for s in all_scores],
        "total_samples": len(y),
        "positive_samples": int(y.sum()),
        "negative_samples": int(len(y) - y.sum()),
    }
    (OUTPUT_DIR / "cv_results.json").write_text(
        json.dumps(results_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"Results saved to {OUTPUT_DIR / 'cv_results.json'}")


if __name__ == "__main__":
    main()
