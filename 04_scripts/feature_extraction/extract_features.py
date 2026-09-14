import zipfile
import pandas as pd
import numpy as np
import os
from scipy import signal
from scipy.stats import iqr

def extract_ppg_features(ppg_signal, fs=100):
    """从PPG信号提取10个特征"""
    features = {}
    
    # 1. 静息心率 (Resting Heart Rate)
    # 使用自相关检测心跳周期
    nyq = fs / 2
    b, a = signal.butter(4, [0.5/nyq, 40/nyq], btype='band')
    ppg_smooth = signal.filtfilt(b, a, ppg_signal)
    peaks, _ = signal.find_peaks(ppg_smooth, distance=int(fs*0.5), height=np.std(ppg_smooth))
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        heart_rate = 60 / np.mean(rr_intervals)
    else:
        heart_rate = 0
    features['resting_hr'] = heart_rate
    
    # 2. 正常心动周期标准差 (SDNN)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        sdnn = np.std(rr_intervals) * 1000  # 转换为ms
    else:
        sdnn = 0
    features['sdnn'] = sdnn
    
    # 3. 相邻间期差均方根 (RMSSD)
    if len(peaks) > 2:
        rr_intervals = np.diff(peaks) / fs
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2)) * 1000
    else:
        rmssd = 0
    features['rmssd'] = rmssd
    
    # 4. 低高频功率比 (LF/HF Ratio)
    if len(ppg_signal) > fs * 10:
        freqs, psd = signal.welch(ppg_signal, fs, nperseg=min(256, len(ppg_signal)))
        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        hf_mask = (freqs > 0.15) & (freqs <= 0.4)
        lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
        hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
    else:
        lf_hf_ratio = 0
    features['lf_hf_ratio'] = lf_hf_ratio
    
    # 5. 灌注指数 (Perfusion Index)
    ac_component = np.std(ppg_signal)
    dc_component = np.mean(np.abs(ppg_signal))
    perfusion_index = (ac_component / dc_component * 100) if dc_component > 0 else 0
    features['perfusion_index'] = perfusion_index
    
    # 6. 增强指数 (Augmentation Index)
    if len(peaks) > 0:
        systolic_peaks = ppg_signal[peaks]
        dicrotic_notch = np.min(ppg_signal[peaks[0]:peaks[-1]]) if len(peaks) > 1 else 0
        augmentation_index = ((np.max(systolic_peaks) - dicrotic_notch) / np.max(systolic_peaks) * 100) if np.max(systolic_peaks) > 0 else 0
    else:
        augmentation_index = 0
    features['augmentation_index'] = augmentation_index
    
    # 7. 主重峰时差 (Peak-to-Peak Time)
    if len(peaks) >= 2:
        peak_to_peak_time = np.mean(np.diff(peaks)) / fs * 1000  # ms
    else:
        peak_to_peak_time = 0
    features['peak_to_peak_time'] = peak_to_peak_time
    
    # 8. ST段偏移量 (ST Segment Deviation)
    if len(peaks) >= 2:
        st_segments = []
        for i in range(len(peaks)-1):
            st_start = peaks[i] + int(fs * 0.08)
            st_end = peaks[i] + int(fs * 0.12)
            if st_end < len(ppg_signal):
                st_segments.append(np.mean(ppg_signal[st_start:st_end]))
        st_deviation = np.mean(st_segments) - np.mean(ppg_signal) if st_segments else 0
    else:
        st_deviation = 0
    features['st_deviation'] = st_deviation
    
    # 9. 校正QT间期 (QTc)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        qt_interval = np.mean(rr_intervals) * 0.4  # 估计QT间期
        qtc = qt_interval / np.sqrt(np.mean(rr_intervals)) * 1000 if np.mean(rr_intervals) > 0 else 0
    else:
        qtc = 0
    features['qtc'] = qtc
    
    # 10. 脉搏波传导时间 (PWV)
    if len(peaks) >= 2:
        pulse_transit_time = np.mean(np.diff(peaks)) / fs
        estimated_velocity = 1 / pulse_transit_time if pulse_transit_time > 0 else 0
    else:
        estimated_velocity = 0
    features['pwv'] = estimated_velocity
    
    return features

def process_single_file(zip_path):
    """处理单个压缩包，提取特征"""
    try:
        z = zipfile.ZipFile(zip_path)
        data_file = [f for f in z.namelist() if f.startswith('collect_data')][0]
        data = z.read(data_file).decode('utf-8')
        lines = data.strip().split('\n')
        header = lines[0].split('\t')
        rows = [line.split('\t') for line in lines[1:] if line.strip()]
        df = pd.DataFrame(rows, columns=header)
        
        # 选择有效的PPG通道
        ppg_cols = ['PPG2', 'PPG3', 'PPG5', 'PPG7', 'PPG9', 'PPG10', 'PPG12', 'PPG13']
        
        # 合并有效通道数据
        ppg_data = []
        for col in ppg_cols:
            if col in df.columns:
                ppg_data.append(pd.to_numeric(df[col], errors='coerce').values)
        
        if not ppg_data:
            return None
        
        # 使用第一个有效通道进行特征提取
        ppg_signal = ppg_data[0]
        
        # 去除零值和异常值
        ppg_signal = ppg_signal[ppg_signal != 0]
        if len(ppg_signal) < 1000:
            return None
        
        # 提取特征
        features = extract_ppg_features(ppg_signal, fs=100)
        
        return features
    except Exception as e:
        print(f"Error processing {zip_path}: {e}")
        return None

def main():
    path = r'E:\data\dataset\attachments\attachments'
    zip_files = [f for f in os.listdir(path) if f.endswith('.zip')]
    
    print(f"总压缩包数量: {len(zip_files)}")
    
    # 处理所有文件
    all_features = []
    for i, zf in enumerate(zip_files[:10]):  # 先测试前10个
        features = process_single_file(os.path.join(path, zf))
        if features:
            features['file'] = zf
            all_features.append(features)
        
        if (i+1) % 10 == 0:
            print(f"已处理 {i+1}/{min(100, len(zip_files))} 个文件")
    
    if not all_features:
        print("未提取到有效特征")
        return
    
    # 转换为DataFrame
    features_df = pd.DataFrame(all_features)
    
    print(f"\n成功提取特征的文件数: {len(features_df)}")
    print(f"\n特征统计:")
    print(features_df.describe())
    
    # 保存特征
    features_df.to_csv(r'E:\data\dataset\ecg_features.csv', index=False)
    print(f"\n特征已保存到: E:\\data\\dataset\\ecg_features.csv")

if __name__ == "__main__":
    main()
