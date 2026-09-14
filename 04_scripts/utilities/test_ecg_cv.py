"""Quick test: 5-fold CV using pre-extracted ECG features (18 samples)."""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

df = pd.read_csv(r'E:\data\dataset\ecg_features_with_labels.csv')
print(f'Samples: {len(df)}, Label dist: {dict(df.label.value_counts())}')

FEATURES = ['resting_hr','sdnn','rmssd','lf_hf_ratio','perfusion_index',
            'augmentation_index','peak_to_peak_time','st_deviation','qtc','pwv']

X = SimpleImputer(strategy='median', add_indicator=True).fit_transform(df[FEATURES])
y = df['label'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_scores = []
for fold, (tr, va) in enumerate(skf.split(X, y)):
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
        print(f'  Fold {fold+1}: skipped'); continue
    m = XGBClassifier(max_depth=4, n_estimators=300, learning_rate=0.04,
                      subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                      random_state=42)
    m.fit(X[tr], y[tr])
    proba = m.predict_proba(X[va])[:,1]
    auc = roc_auc_score(y[va], proba)
    all_scores.append(auc)
    preds = (proba >= 0.5).astype(int)
    print(f'  Fold {fold+1}: AUC={auc:.4f}, n={len(y[va])}, pos={int(y[va].sum())}')

mean_auc = np.mean(all_scores); std_auc = np.std(all_scores)
print(f'\n5-Fold CV Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})')
print(f'Folds: {[f"{s:.4f}" for s in all_scores]}')
print(f'N={len(y)}, pos={int(y.sum())}, neg={int(len(y)-y.sum())}')
