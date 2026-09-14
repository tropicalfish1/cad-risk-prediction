"""五折分层交叉验证（改进版）

改进点:
1. KFold -> StratifiedKFold，保证每折类别比例一致
2. 手动标准化 -> Pipeline，防止数据泄露
3. 添加 class_weight='balanced'，处理类别不平衡
4. 添加 Precision/Recall/F1 评估指标
5. 特征缓存，避免重复解压提取
"""
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['resting_hr_bpm', 'sdnn_ms', 'rmssd_ms', 'lf_hf_ratio', 'perfusion_index',
            'augmentation_index', 'primary_reflected_peak_delay_ms', 'st_deviation_mv',
            'qtc_ms', 'pulse_transit_time_ms']

ATTACH_DIR = Path(r'E:\data\dataset\attachments\attachments')
MAPPING_FILE = Path(r'E:\data\dataset\attachments\标签对应表.txt')
OUTPUT_DIR = Path(r'E:\data\dataset')
CACHE_FILE = Path(r'E:\data\attachments_cv_output\extracted_features.csv')


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def extract_all_features():
    if CACHE_FILE.exists():
        log(f'Found cached features: {CACHE_FILE}')
        df = pd.read_csv(CACHE_FILE)
        log(f'Loaded {len(df)} cached samples')
        return df

    log('ERROR: No cached features found! Run attachments_5fold_cv.py first to extract features.')
    return None


def run_cross_validation(df):
    log(f'Label distribution: {dict(df["label"].value_counts())}')

    X = SimpleImputer(strategy='median').fit_transform(df[FEATURES])
    y = df['label'].values

    log(f'Feature matrix: {X.shape}')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_list, auc_list, prec_list, rec_list, f1_list = [], [], [], [], []

    log('\n' + '=' * 60)
    log('5-Fold Stratified Cross-Validation')
    log('=' * 60)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        try:
            auc = roc_auc_score(y_val, y_prob)
        except ValueError:
            auc = 0.0
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        acc_list.append(acc)
        auc_list.append(auc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

        log(f'\nFold {fold + 1}:')
        log(f'  Accuracy  = {acc:.4f}')
        log(f'  AUC       = {auc:.4f}')
        log(f'  Precision = {prec:.4f}')
        log(f'  Recall    = {rec:.4f}')
        log(f'  F1        = {f1:.4f}')
        log(f'  Train: {len(y_train)} (pos={int(y_train.sum())}, neg={int(len(y_train)-y_train.sum())})')
        log(f'  Val:   {len(y_val)} (pos={int(y_val.sum())}, neg={int(len(y_val)-y_val.sum())})')

    log('\n' + '=' * 60)
    log('Average Results:')
    log(f'  Accuracy:  {np.mean(acc_list):.4f} (+/- {np.std(acc_list):.4f})')
    valid_auc = [a for a in auc_list if a > 0]
    if valid_auc:
        log(f'  AUC:       {np.mean(valid_auc):.4f} (+/- {np.std(valid_auc):.4f})')
    log(f'  Precision: {np.mean(prec_list):.4f} (+/- {np.std(prec_list):.4f})')
    log(f'  Recall:    {np.mean(rec_list):.4f} (+/- {np.std(rec_list):.4f})')
    log(f'  F1:        {np.mean(f1_list):.4f} (+/- {np.std(f1_list):.4f})')
    log(f'  Total samples: {len(y)} (pos={int(y.sum())}, neg={int(len(y)-y.sum())})')
    log('=' * 60)

    log('\nFeature Importance (full-data training):')
    pipeline_final = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=6,
                                       class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    pipeline_final.fit(X, y)
    imp = pd.DataFrame({
        'feature': FEATURES,
        'importance': pipeline_final.named_steps['clf'].feature_importances_
    }).sort_values('importance', ascending=False)
    log(imp.to_string(index=False))

    summary = {
        'mean_accuracy': float(np.mean(acc_list)),
        'mean_auc': float(np.mean(valid_auc)) if valid_auc else 0,
        'mean_precision': float(np.mean(prec_list)),
        'mean_recall': float(np.mean(rec_list)),
        'mean_f1': float(np.mean(f1_list)),
        'std_accuracy': float(np.std(acc_list)),
        'std_auc': float(np.std(valid_auc)) if valid_auc else 0,
        'fold_aucs': [float(a) for a in auc_list],
        'total_samples': int(len(y)),
        'positive_samples': int(y.sum()),
    }
    (OUTPUT_DIR / 'cv_results_improved.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f'\nResults saved to {OUTPUT_DIR / "cv_results_improved.json"}')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = extract_all_features()
    if df is None or len(df) == 0:
        log('ERROR: No features extracted!')
        return
    run_cross_validation(df)


if __name__ == '__main__':
    main()
