import zipfile
import pandas as pd
import numpy as np
import os

path = r'E:\data\dataset\attachments\attachments'
zip_files = [f for f in os.listdir(path) if f.endswith('.zip')][:10]

print('=== 多文件数据质量对比 ===')
results = []

for zf in zip_files:
    try:
        z = zipfile.ZipFile(os.path.join(path, zf))
        data_file = [f for f in z.namelist() if f.startswith('collect_data')][0]
        data = z.read(data_file).decode('utf-8')
        lines = data.strip().split('\n')
        header = lines[0].split('\t')
        rows = [line.split('\t') for line in lines[1:] if line.strip()]
        df = pd.DataFrame(rows, columns=header)

        ppg_cols = [col for col in df.columns if col.startswith('PPG') and col != 'PPG_TIME']
        valid_channels = 0
        for col in ppg_cols:
            col_data = pd.to_numeric(df[col], errors='coerce')
            if (col_data != 0).sum() > 1000:
                valid_channels += 1

        results.append({
            'file': zf[:20] + '...',
            'rows': len(df),
            'valid_ppg': valid_channels,
            'total_ppg': len(ppg_cols)
        })
    except Exception as e:
        print(f'Error processing {zf}: {e}')

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print('\n=== 总体评估 ===')
avg_valid = results_df['valid_ppg'].mean()
avg_rows = results_df['rows'].mean()
print(f'平均有效PPG通道数: {avg_valid:.1f}')
print(f'平均数据行数: {avg_rows:.0f}')
print(f'数据量充足性: {"充足" if avg_rows > 100000 else "不足"}')

total_zips = len([f for f in os.listdir(path) if f.endswith('.zip')])
print(f'\n总压缩包数量: {total_zips}')
