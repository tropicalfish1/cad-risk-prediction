import zipfile
import pandas as pd
import numpy as np
import os
import json
from scipy import signal
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import time

def extract_ppg_features(ppg_signal, fs=100):
    """从PPG信号提取10个特征"""
    features = {}
    
    if len(ppg_signal) > 1000:
        indices = np.linspace(0, len(ppg_signal)-1, 1000, dtype=int)
        ppg_signal = ppg_signal[indices]
    
    ppg_signal = ppg_signal[ppg_signal != 0]
    if len(ppg_signal) < 500:
        return None
    
    nyq = fs / 2
    b, a = signal.butter(4, [0.5/nyq, 40/nyq], btype='band')
    ppg_smooth = signal.filtfilt(b, a, ppg_signal)
    
    peaks, _ = signal.find_peaks(ppg_smooth, distance=int(fs*0.5), height=np.std(ppg_smooth))
    
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        features['resting_hr'] = 60 / np.mean(rr_intervals)
        features['sdnn'] = np.std(rr_intervals) * 1000
    else:
        features['resting_hr'] = 0
        features['sdnn'] = 0
    
    if len(peaks) > 2:
        rr_intervals = np.diff(peaks) / fs
        features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2)) * 1000
    else:
        features['rmssd'] = 0
    
    if len(ppg_signal) > fs * 5:
        freqs, psd = signal.welch(ppg_signal, fs, nperseg=min(256, len(ppg_signal)))
        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        hf_mask = (freqs > 0.15) & (freqs <= 0.4)
        lf_power = np.trapezoid(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
        hf_power = np.trapezoid(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
        features['lf_hf_ratio'] = lf_power / hf_power if hf_power > 0 else 0
    else:
        features['lf_hf_ratio'] = 0
    
    ac_component = np.std(ppg_signal)
    dc_component = np.mean(np.abs(ppg_signal))
    features['perfusion_index'] = (ac_component / dc_component * 100) if dc_component > 0 else 0
    
    if len(peaks) > 0:
        systolic_peaks = ppg_signal[peaks]
        dicrotic_notch = np.min(ppg_signal[peaks[0]:peaks[-1]]) if len(peaks) > 1 else 0
        features['augmentation_index'] = ((np.max(systolic_peaks) - dicrotic_notch) / np.max(systolic_peaks) * 100) if np.max(systolic_peaks) > 0 else 0
    else:
        features['augmentation_index'] = 0
    
    if len(peaks) >= 2:
        features['peak_to_peak_time'] = np.mean(np.diff(peaks)) / fs * 1000
    else:
        features['peak_to_peak_time'] = 0
    
    if len(peaks) >= 2:
        st_segments = []
        for i in range(min(len(peaks)-1, 10)):
            st_start = peaks[i] + int(fs * 0.08)
            st_end = peaks[i] + int(fs * 0.12)
            if st_end < len(ppg_signal):
                st_segments.append(np.mean(ppg_signal[st_start:st_end]))
        features['st_deviation'] = np.mean(st_segments) - np.mean(ppg_signal) if st_segments else 0
    else:
        features['st_deviation'] = 0
    
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        qt_interval = np.mean(rr_intervals) * 0.4
        features['qtc'] = qt_interval / np.sqrt(np.mean(rr_intervals)) * 1000 if np.mean(rr_intervals) > 0 else 0
    else:
        features['qtc'] = 0
    
    if len(peaks) >= 2:
        pulse_transit_time = np.mean(np.diff(peaks)) / fs
        features['pwv'] = 1 / pulse_transit_time if pulse_transit_time > 0 else 0
    else:
        features['pwv'] = 0
    
    return features

def process_single_file(zip_path):
    """处理单个压缩包"""
    try:
        z = zipfile.ZipFile(zip_path)
        data_file = [f for f in z.namelist() if f.startswith('collect_data')][0]
        data = z.read(data_file).decode('utf-8')
        lines = data.strip().split('\n')
        header = lines[0].split('\t')
        rows = [line.split('\t') for line in lines[1:] if line.strip()]
        df = pd.DataFrame(rows, columns=header)
        
        ppg_cols = ['PPG2', 'PPG3', 'PPG5', 'PPG7', 'PPG9', 'PPG10', 'PPG12', 'PPG13']
        ppg_data = []
        for col in ppg_cols:
            if col in df.columns:
                ppg_data.append(pd.to_numeric(df[col], errors='coerce').values)
        
        if not ppg_data:
            return None
        
        ppg_signal = ppg_data[0]
        features = extract_ppg_features(ppg_signal, fs=100)
        return features
    except:
        return None

def main():
    path = r'E:\data\dataset\attachments\attachments'
    zip_files = [f for f in os.listdir(path) if f.endswith('.zip')]
    
    print("总压缩包数量: %d" % len(zip_files))
    start_time = time.time()
    
    # 处理所有文件
    all_features = []
    file_names = []
    
    for i, zf in enumerate(zip_files):
        features = process_single_file(os.path.join(path, zf))
        if features:
            all_features.append(features)
            file_names.append(zf)
        
        if (i+1) % 100 == 0:
            elapsed = time.time() - start_time
            print("已处理 %d/%d 个文件, 耗时: %.1f秒, 成功: %d" % (i+1, len(zip_files), elapsed, len(all_features)))
    
    print("成功提取特征的文件数: %d" % len(all_features))
    print("总耗时: %.1f秒" % (time.time() - start_time))
    
    if not all_features:
        print("未提取到有效特征")
        return
    
    # 转换为DataFrame
    features_df = pd.DataFrame(all_features)
    features_df['file'] = file_names
    
    # 创建模拟标签
    np.random.seed(42)
    labels = np.random.choice([0, 1], size=len(features_df), p=[0.6, 0.4])
    features_df['label'] = labels
    
    # 准备特征矩阵
    feature_cols = ['resting_hr', 'sdnn', 'rmssd', 'lf_hf_ratio', 'perfusion_index', 
                   'augmentation_index', 'peak_to_peak_time', 'st_deviation', 'qtc', 'pwv']
    
    X = features_df[feature_cols].values
    y = features_df['label'].values
    
    print("\n最终样本分布:")
    print("正样本: %d" % np.sum(y == 1))
    print("负样本: %d" % np.sum(y == 0))
    
    # 五折交叉验证
    print("\n=== 五折交叉验证 ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracy_scores = []
    auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if len(np.unique(y_val)) < 2:
            print("Fold %d: 跳过（验证集只有一个类别）" % (fold+1))
            continue
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_val_scaled)
        y_prob = model.predict_proba(X_val_scaled)[:, 1]
        
        accuracy = accuracy_score(y_val, y_pred)
        try:
            auc = roc_auc_score(y_val, y_prob)
        except:
            auc = 0
        
        accuracy_scores.append(accuracy)
        auc_scores.append(auc)
        
        print("Fold %d: Accuracy=%.4f, AUC=%.4f" % (fold+1, accuracy, auc))
    
    if accuracy_scores:
        print("\n=== 平均结果 ===")
        print("平均准确率: %.4f (+/- %.4f)" % (np.mean(accuracy_scores), np.std(accuracy_scores)))
        if auc_scores:
            valid_auc = [a for a in auc_scores if a > 0]
            if valid_auc:
                print("平均AUC: %.4f (+/- %.4f)" % (np.mean(valid_auc), np.std(valid_auc)))
    
    # 特征重要性
    print("\n=== 特征重要性 ===")
    model_final = RandomForestClassifier(n_estimators=100, random_state=42)
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    model_final.fit(X_scaled, y)
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model_final.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(importance_df.to_string(index=False))
    
    # 保存结果
    features_df.to_csv(r'E:\data\dataset\ecg_features_all.csv', index=False)
    print("\n结果已保存到: E:\\data\\dataset\\ecg_features_all.csv")

if __name__ == "__main__":
    main()
