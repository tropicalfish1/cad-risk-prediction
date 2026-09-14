# CAD（冠心病）风险预测模型项目

基于华为手表PPG/ECG信号的冠心病风险预测模型，使用XGBoost算法构建二分类模型。

## 项目概述

本项目利用华为手表采集的光电容积脉搏波(PPG)和心电(ECG)信号，提取10项生理特征，通过机器学习算法（XGBoost）构建冠心病风险预测模型。

### 主要成果
- **模型AUC**: 0.8101 (改进版)
- **训练样本**: 106,973例
- **验证样本**: 909例 (37例正样本, 872例负样本)
- **特征数量**: 10项PPG特征

## 目录结构

```
data/
├── 01_raw_data/                    # 原始数据
│   ├── huaweidata/                 # 华为原始JSONL数据
│   │   ├── userbasicinfo/          # 用户基本信息
│   │   ├── vascularacc/            # 血管问卷数据
│   │   ├── vascularecg/            # 血管ECG数据
│   │   ├── vascularppg/            # 血管PPG数据
│   │   └── public_ppg_ecg_pretrain_clean_v1.zip
│   └── attachments/                # ZIP附件原始数据
│       ├── attachments/            # 1000个ZIP文件
│       └── 标签对应表.txt           # 标签映射表
│
├── 02_feature_data/                # 特征与结构化数据
│   ├── structured/                 # 结构化数据集
│   │   ├── attachment_*.csv        # 附件相关数据
│   │   ├── patient_map*.csv        # 患者映射
│   │   ├── feature_dictionary.csv  # 特征字典
│   │   ├── ml_features.csv         # ML特征数据
│   │   └── norm_*.csv              # 标准化数据
│   ├── attachments_wave/           # npz波形数据
│   ├── ecg_features*.csv           # ECG特征数据集
│   └── extracted_features.csv      # 附件提取的909样本特征
│
├── 03_models/                      # 模型文件
│   ├── cad_xgboost.json            # XGBoost主模型
│   └── cad_xgboost_ptt.json        # PTT增强模型
│
├── 04_scripts/                     # Python脚本
│   ├── training/                   # 训练脚本
│   │   ├── cad_xgboost_pipeline.py # 主训练管线
│   │   └── cad_inference.py        # 推理脚本
│   ├── cross_validation/           # 交叉验证脚本
│   │   ├── attachments_5fold_cv.py # 完整版CV
│   │   ├── attachments_cv_fast.py  # 快速版CV
│   │   └── cross_validation_*.py   # 历史迭代版本
│   ├── feature_extraction/         # 特征提取
│   │   └── extract_features.py
│   ├── report_generation/          # 报告生成
│   │   ├── gen_report_ppt.py
│   │   ├── create_ppt.py
│   │   ├── create_comparison_ppt.py
│   │   └── create_baseline_comparison_ppt.py
│   └── utilities/                  # 工具脚本
│       ├── check_ecg.py
│       ├── test_ecg_cv.py
│       └── check_quality.py
│
├── 05_results/                     # 实验结果
│   ├── cad_model_output/           # 主管线输出
│   ├── cad_model_output_ptt/       # PTT增强输出
│   ├── attachments_cv_output/      # 附件CV结果
│   ├── test_results/               # 测试结果
│   └── structured_results/         # 结构化结果
│       ├── splits/                 # 数据划分结果
│       └── splits_chd/             # CHD模型结果
│
├── 06_reports/                     # 报告与文档
│   ├── CAD_Risk_Prediction_Model_Report_Chinese_v2.pptx
│   ├── Baseline_vs_改进版对比.pptx
│   ├── 交叉验证改进对比.pptx
│   └── CAD_basline_cv_report(1).pptx
│
├── 07_pretrain/                    # 预训练相关
│   ├── pretrain_config.json        # 预训练配置
│   └── pretrained_encoder_*.pth    # 预训练编码器权重
│
├── README.md                       # 项目说明（本文件）
├── 项目流程文档.md                  # 详细项目文档
└── .venv/                          # Python虚拟环境
```

## 核心文件说明

### 训练脚本
- **`04_scripts/training/cad_xgboost_pipeline.py`**: 主训练管线，从问卷标签和PPG/ECG附件中提取特征，训练XGBoost模型
- **`04_scripts/training/cad_inference.py`**: 前瞻性推理脚本，加载训练好的模型进行预测

### 交叉验证
- **`04_scripts/cross_validation/attachments_5fold_cv.py`**: 5折交叉验证完整版
- **`04_scripts/cross_validation/attachments_cv_fast.py`**: 快速5折CV（仅处理前500个文件）

### 特征工程
提取的10项PPG特征：
1. **静息心率 (resting_hr_bpm)**: 反映基础心血管状态
2. **SDNN (sdnn_ms)**: 心动周期标准差，反映自主神经整体变异
3. **RMSSD (rmssd_ms)**: 相邻间期差均方根，反映副交感神经活性
4. **LF/HF比 (lf_hf_ratio)**: 低高频功率比，反映自主神经平衡
5. **灌注指数 (perfusion_index)**: 反映末梢血液循环
6. **增强指数 (augmentation_index)**: 反映动脉僵硬度
7. **反射峰延迟 (primary_reflected_peak_delay_ms)**: 反映动脉硬化程度
8. **ST段偏移 (st_deviation_mv)**: 反映心肌缺血状态
9. **校正QT间期 (qtc_ms)**: 反映心室复极时间
10. **脉搏传导时间 (pulse_transit_time_ms)**: 反映血压变化

## 模型性能

### 五折交叉验证结果 (改进版)
| 指标 | 值 |
|------|-----|
| Mean AUC | 0.8101 ± 0.043 |
| Accuracy | 0.9043 ± 0.028 |
| Precision | 0.2085 ± 0.097 |
| Recall | 0.4071 ± 0.094 |
| F1 Score | 0.2705 ± 0.100 |

### 特征重要性排名
1. ST段偏移 (st_deviation_mv): 0.196
2. 灌注指数 (perfusion_index): 0.151
3. RMSSD (rmssd_ms): 0.138
4. SDNN (sdnn_ms): 0.099
5. 反射峰延迟 (primary_reflected_peak_delay_ms): 0.091

## 使用方法

### 1. 环境配置
```bash
# 激活虚拟环境
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备
- 将华为原始数据放入 `01_raw_data/huaweidata/`
- 将ZIP附件放入 `01_raw_data/attachments/attachments/`

### 3. 模型训练
```bash
# 运行主训练管线
python 04_scripts/training/cad_xgboost_pipeline.py
```

### 4. 交叉验证
```bash
# 运行5折交叉验证
python 04_scripts/cross_validation/attachments_5fold_cv.py
```

### 5. 推理预测
```bash
# 运行推理脚本
python 04_scripts/training/cad_inference.py
```

## 项目改进历程

### 主要改进点
1. **数据泄露防护**: 使用Pipeline封装，标准化仅在训练集fit
2. **类别不平衡处理**: 添加class_weight='balanced'自动调整权重
3. **交叉验证优化**: KFold → StratifiedKFold，保证每折类别比例一致
4. **评估指标完善**: 新增Precision/Recall/F1，全面评估模型性能

### 改进前后对比
| 指标 | Baseline | 改进版 | 提升 |
|------|---------|--------|------|
| AUC | 0.8043 | 0.8101 | +0.7% |
| Accuracy | - | 0.9043 | 新增 |
| Recall | - | 0.4071 | 新增 |

## 后续优化方向

1. **数据层面**: SMOTE/ADASYN过采样正样本，收集更多CAD+数据
2. **模型层面**: 尝试XGBoost/LightGBM + class_weight，贝叶斯超参优化
3. **阈值优化**: 当前0.5 → 调整为0.3以提升Recall
4. **特征工程**: ST段偏移和灌注指数为最重要特征，可重点优化

## 相关文档

- **项目流程文档.md**: 详细项目文档，包含数据集来源、特征工程、模型架构、性能指标
- **06_reports/**: 包含所有PPT报告文件

## 注意事项

1. **数据安全**: 原始数据和模型文件较大，建议使用Git LFS管理
2. **虚拟环境**: `.venv/` 目录不应提交到版本控制
3. **类别不平衡**: 正负样本比例为1:23.6，需特别关注Recall指标
4. **特征缺失**: 3个ECG特征全部缺失，当前仅使用7个有效PPG特征

## 技术栈

- **Python**: 3.12
- **机器学习**: XGBoost, Scikit-learn
- **信号处理**: SciPy
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib (用于报告生成)
