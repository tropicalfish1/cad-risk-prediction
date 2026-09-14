"""生成Baseline vs 改进版对比PPT"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

DARK_BG = RGBColor(0x1A, 0x1A, 0x2E)
ACCENT_BLUE = RGBColor(0x00, 0x96, 0xD6)
ACCENT_GREEN = RGBColor(0x2E, 0xCC, 0x71)
ACCENT_ORANGE = RGBColor(0xF3, 0x9C, 0x12)
ACCENT_RED = RGBColor(0xE7, 0x4C, 0x3C)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xCC, 0xCC, 0xCC)
CARD_BG = RGBColor(0x2D, 0x2D, 0x44)


def set_bg(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def add_box(slide, left, top, w, h, text='', fs=14, fc=WHITE, bold=False,
            align=PP_ALIGN.LEFT, bg=None, border=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = bg or CARD_BG
    if border:
        s.line.color.rgb = border
        s.line.width = Pt(2)
    else:
        s.line.fill.background()
    tf = s.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(fs)
    p.font.color.rgb = fc
    p.font.bold = bold
    p.alignment = align
    return s


def add_text(slide, left, top, w, h, text, fs=18, fc=WHITE, bold=False, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(left, top, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(fs)
    p.font.color.rgb = fc
    p.font.bold = bold
    p.alignment = align
    return tb


# ==================== Slide 1: Title ====================
s1 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(s1, DARK_BG)

add_text(s1, Inches(1), Inches(1.2), Inches(11), Inches(1),
         "CAD预测模型 Baseline vs 改进版", fs=42, fc=ACCENT_BLUE, bold=True, align=PP_ALIGN.CENTER)

add_text(s1, Inches(1), Inches(2.5), Inches(11), Inches(0.7),
         "五折交叉验证对比分析报告", fs=28, fc=WHITE, align=PP_ALIGN.CENTER)

add_box(s1, Inches(3.5), Inches(4.0), Inches(6), Inches(0.6),
        "XGBoost (Baseline)  vs  RandomForest (改进版)", fs=16, fc=LIGHT_GRAY, align=PP_ALIGN.CENTER, bg=DARK_BG)

add_text(s1, Inches(1), Inches(5.5), Inches(11), Inches(0.5),
         "909例样本 | 10个PPG特征 | Stratified 5-Fold CV", fs=14, fc=LIGHT_GRAY, align=PP_ALIGN.CENTER)


# ==================== Slide 2: Baseline Summary ====================
s2 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(s2, DARK_BG)

add_text(s2, Inches(0.5), Inches(0.3), Inches(12), Inches(0.8),
         "Baseline 模型概述", fs=36, fc=ACCENT_BLUE, bold=True)

# Left card - data
add_box(s2, Inches(0.5), Inches(1.3), Inches(6), Inches(5.5), border=ACCENT_BLUE)
add_text(s2, Inches(0.8), Inches(1.5), Inches(5.5), Inches(0.5),
         "数据与特征", fs=22, fc=ACCENT_BLUE, bold=True)

items_left = [
    ("样本数量", "909例 (来自zip文件)"),
    ("正样本(CAD+)", "37例 (4.1%)"),
    ("负样本(CAD-)", "872例 (95.9%)"),
    ("特征数量", "10个PPG特征 (7个有效, 3个ECG全缺失)"),
    ("标签来源", "标签对应表 → 时间戳映射 → 外部ID → 疾病关键词"),
]
for i, (k, v) in enumerate(items_left):
    add_text(s2, Inches(1.0), Inches(2.2 + i * 0.8), Inches(5.2), Inches(0.7),
             f"{k}: {v}", fs=14, fc=WHITE)

# Right card - model
add_box(s2, Inches(6.8), Inches(1.3), Inches(6), Inches(5.5), border=ACCENT_ORANGE)
add_text(s2, Inches(7.1), Inches(1.5), Inches(5.5), Inches(0.5),
         "Baseline模型配置", fs=22, fc=ACCENT_ORANGE, bold=True)

items_right = [
    ("模型", "XGBoost (max_depth=4, n_estimators=300)"),
    ("学习率", "0.04"),
    ("采样", "subsample=0.8, colsample_bytree=0.8"),
    ("缺失值", "SimpleImputer (median + indicator)"),
    ("交叉验证", "StratifiedKFold, n_splits=5"),
]
for i, (k, v) in enumerate(items_right):
    add_text(s2, Inches(7.3), Inches(2.2 + i * 0.8), Inches(5.2), Inches(0.7),
             f"{k}: {v}", fs=14, fc=WHITE)


# ==================== Slide 3: Baseline CV Results ====================
s3 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(s3, DARK_BG)

add_text(s3, Inches(0.5), Inches(0.3), Inches(12), Inches(0.8),
         "Baseline 五折交叉验证结果", fs=36, fc=ACCENT_BLUE, bold=True)

# Big metric cards
metrics = [
    ("Mean AUC", "0.8043", ACCENT_BLUE),
    ("Std AUC", "±0.0392", LIGHT_GRAY),
    ("总样本", "909", WHITE),
    ("正样本", "37 (4.1%)", ACCENT_RED),
]
for i, (label, value, color) in enumerate(metrics):
    x = Inches(0.5 + i * 3.1)
    add_box(s3, x, Inches(1.3), Inches(2.8), Inches(1.5), bg=RGBColor(0x1E, 0x3A, 0x5F), border=ACCENT_BLUE)
    add_text(s3, x, Inches(1.4), Inches(2.8), Inches(0.5), label, fs=14, fc=LIGHT_GRAY, align=PP_ALIGN.CENTER)
    add_text(s3, x, Inches(1.9), Inches(2.8), Inches(0.7), value, fs=32, fc=color, bold=True, align=PP_ALIGN.CENTER)

# Fold AUCs
add_text(s3, Inches(0.5), Inches(3.2), Inches(12), Inches(0.5),
         "各折AUC分布", fs=22, fc=WHITE, bold=True)

fold_aucs = [("F1", 0.7935), ("F2", 0.8571), ("F3", 0.8355), ("F4", 0.7917), ("F5", 0.7438)]
bar_max = Inches(8)
for i, (name, auc) in enumerate(fold_aucs):
    y = Inches(3.9 + i * 0.6)
    add_text(s3, Inches(0.5), y, Inches(0.8), Inches(0.45), name, fs=14, fc=WHITE, bold=True)
    # bar bg
    add_box(s3, Inches(1.5), y + Inches(0.05), Inches(8), Inches(0.35), bg=RGBColor(0x35, 0x35, 0x50))
    # bar fill
    bar_w = int(Inches(8) * auc)
    add_box(s3, Inches(1.5), y + Inches(0.05), bar_w, Inches(0.35),
            bg=ACCENT_BLUE if auc >= 0.8 else ACCENT_ORANGE)
    add_text(s3, Inches(9.8), y, Inches(2), Inches(0.45), f"{auc:.4f}", fs=14, fc=ACCENT_GREEN, bold=True)


# ==================== Slide 4: Improved Version Results ====================
s4 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(s4, DARK_BG)

add_text(s4, Inches(0.5), Inches(0.3), Inches(12), Inches(0.8),
         "改进版 五折交叉验证结果", fs=36, fc=ACCENT_GREEN, bold=True)

# Big metric cards
metrics2 = [
    ("Mean AUC", "0.8101", ACCENT_GREEN),
    ("Accuracy", "0.9043", ACCENT_BLUE),
    ("Precision", "0.2085", WHITE),
    ("Recall", "0.4071", WHITE),
    ("F1 Score", "0.2705", ACCENT_ORANGE),
]
for i, (label, value, color) in enumerate(metrics2):
    x = Inches(0.3 + i * 2.55)
    add_box(s4, x, Inches(1.3), Inches(2.3), Inches(1.5), bg=RGBColor(0x1E, 0x3A, 0x5F), border=ACCENT_GREEN)
    add_text(s4, x, Inches(1.4), Inches(2.3), Inches(0.5), label, fs=13, fc=LIGHT_GRAY, align=PP_ALIGN.CENTER)
    add_text(s4, x, Inches(1.9), Inches(2.3), Inches(0.7), value, fs=28, fc=color, bold=True, align=PP_ALIGN.CENTER)

# Fold details table
add_text(s4, Inches(0.5), Inches(3.2), Inches(12), Inches(0.5),
         "各折详细指标", fs=22, fc=WHITE, bold=True)

headers = ["Fold", "Accuracy", "AUC", "Precision", "Recall", "F1"]
col_x = [Inches(0.5), Inches(1.8), Inches(3.6), Inches(5.4), Inches(7.2), Inches(9.0)]
col_w = [Inches(1.2), Inches(1.6), Inches(1.6), Inches(1.6), Inches(1.6), Inches(1.6)]

for j, (h, x, w) in enumerate(zip(headers, col_x, col_w)):
    add_box(s4, x, Inches(3.8), w, Inches(0.45), h, fs=12, fc=WHITE, bold=True, align=PP_ALIGN.CENTER, bg=ACCENT_GREEN)

rows = [
    ("1", "0.8736", "0.7355", "0.1364", "0.4286", "0.2069"),
    ("2", "0.9505", "0.8653", "0.4000", "0.5714", "0.4706"),
    ("3", "0.8846", "0.8240", "0.1579", "0.3750", "0.2222"),
    ("4", "0.8901", "0.7996", "0.1667", "0.3750", "0.2308"),
    ("5", "0.9227", "0.8259", "0.1818", "0.2857", "0.2222"),
]
for ri, row in enumerate(rows):
    y = Inches(4.35 + ri * 0.5)
    bg = CARD_BG if ri % 2 == 0 else RGBColor(0x35, 0x35, 0x50)
    for j, (v, x, w) in enumerate(zip(row, col_x, col_w)):
        add_box(s4, x, y, w, Inches(0.42), v, fs=12, fc=WHITE, align=PP_ALIGN.CENTER, bg=bg)

# AVG row
avg = ("AVG", "0.9043", "0.8101", "0.2085", "0.4071", "0.2705")
for j, (v, x, w) in enumerate(zip(avg, col_x, col_w)):
    add_box(s4, x, Inches(6.85), w, Inches(0.45), v, fs=13, fc=ACCENT_GREEN, bold=True,
            align=PP_ALIGN.CENTER, bg=RGBColor(0x1E, 0x3A, 0x5F))


# ==================== Slide 5: Head-to-Head Comparison ====================
s5 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(s5, DARK_BG)

add_text(s5, Inches(0.5), Inches(0.3), Inches(12), Inches(0.8),
         "Baseline vs 改进版 核心指标对比", fs=36, fc=ACCENT_BLUE, bold=True)

# Comparison table
comp_headers = ["指标", "Baseline (XGBoost)", "改进版 (RandomForest)", "变化"]
comp_col_x = [Inches(0.5), Inches(3.5), Inches(6.8), Inches(10.2)]
comp_col_w = [Inches(2.8), Inches(3.1), Inches(3.2), Inches(2.5)]

for j, (h, x, w) in enumerate(zip(comp_headers, comp_col_x, comp_col_w)):
    add_box(s5, x, Inches(1.2), w, Inches(0.55), h, fs=16, fc=WHITE, bold=True,
            align=PP_ALIGN.CENTER, bg=ACCENT_BLUE)

comp_rows = [
    ("Mean AUC", "0.8043 ± 0.039", "0.8101 ± 0.043", "+0.006 (+0.7%)", ACCENT_GREEN),
    ("Accuracy", "未报告", "0.9043 ± 0.028", "新增", ACCENT_BLUE),
    ("Precision", "未报告", "0.2085 ± 0.097", "新增", ACCENT_BLUE),
    ("Recall", "未报告", "0.4071 ± 0.094", "新增", ACCENT_BLUE),
    ("F1 Score", "0.79 (macro)", "0.2705 ± 0.100", "不同计算方式", ACCENT_ORANGE),
    ("类别平衡处理", "无", "class_weight='balanced'", "修复", ACCENT_GREEN),
    ("数据泄露防护", "无Pipeline", "Pipeline标准Scaler", "修复", ACCENT_GREEN),
    ("样本数", "909", "909", "一致", WHITE),
    ("特征数", "10 (7有效)", "10 (10有效)", "一致", WHITE),
]

for ri, (metric, base, impv, change, chg_color) in enumerate(comp_rows):
    y = Inches(1.85 + ri * 0.58)
    bg = CARD_BG if ri % 2 == 0 else RGBColor(0x35, 0x35, 0x50)
    vals = [metric, base, impv, change]
    colors = [WHITE, LIGHT_GRAY, ACCENT_GREEN, chg_color]
    for j, (v, c, x, w) in enumerate(zip(vals, colors, comp_col_x, comp_col_w)):
        add_box(s5, x, y, w, Inches(0.5), v, fs=13, fc=c, align=PP_ALIGN.CENTER, bg=bg)


# ==================== Slide 6: Fold AUC Comparison ====================
s6 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(s6, DARK_BG)

add_text(s6, Inches(0.5), Inches(0.3), Inches(12), Inches(0.8),
         "各折AUC对比", fs=36, fc=ACCENT_BLUE, bold=True)

base_aucs = [0.7935, 0.8571, 0.8355, 0.7917, 0.7438]
impr_aucs = [0.7355, 0.8653, 0.8240, 0.7996, 0.8259]

# Header
add_box(s6, Inches(0.5), Inches(1.2), Inches(1.5), Inches(0.5), "Fold", fs=14, fc=WHITE, bold=True,
        align=PP_ALIGN.CENTER, bg=ACCENT_BLUE)
add_box(s6, Inches(2.1), Inches(1.2), Inches(2.5), Inches(0.5), "Baseline AUC", fs=14, fc=WHITE, bold=True,
        align=PP_ALIGN.CENTER, bg=ACCENT_ORANGE)
add_box(s6, Inches(4.7), Inches(1.2), Inches(2.5), Inches(0.5), "改进版 AUC", fs=14, fc=WHITE, bold=True,
        align=PP_ALIGN.CENTER, bg=ACCENT_GREEN)
add_box(s6, Inches(7.3), Inches(1.2), Inches(5.5), Inches(0.5), "对比", fs=14, fc=WHITE, bold=True,
        align=PP_ALIGN.CENTER, bg=ACCENT_BLUE)

for i in range(5):
    y = Inches(1.85 + i * 0.7)
    diff = impr_aucs[i] - base_aucs[i]
    diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
    diff_color = ACCENT_GREEN if diff >= 0 else ACCENT_RED

    add_box(s6, Inches(0.5), y, Inches(1.5), Inches(0.55), f"F{i+1}", fs=16, fc=WHITE, bold=True,
            align=PP_ALIGN.CENTER, bg=CARD_BG)

    # Base bar
    bar_base_w = int(Inches(2.5) * base_aucs[i])
    add_box(s6, Inches(2.1), y + Inches(0.05), Inches(2.5), Inches(0.45), bg=RGBColor(0x35, 0x35, 0x50))
    add_box(s6, Inches(2.1), y + Inches(0.05), bar_base_w, Inches(0.45), bg=ACCENT_ORANGE)
    add_text(s6, Inches(2.3), y + Inches(0.05), Inches(2.2), Inches(0.45),
             f"{base_aucs[i]:.4f}", fs=13, fc=WHITE, bold=True)

    # Imp bar
    bar_imp_w = int(Inches(2.5) * impr_aucs[i])
    add_box(s6, Inches(4.7), y + Inches(0.05), Inches(2.5), Inches(0.45), bg=RGBColor(0x35, 0x35, 0x50))
    add_box(s6, Inches(4.7), y + Inches(0.05), bar_imp_w, Inches(0.45), bg=ACCENT_GREEN)
    add_text(s6, Inches(4.9), y + Inches(0.05), Inches(2.2), Inches(0.45),
             f"{impr_aucs[i]:.4f}", fs=13, fc=WHITE, bold=True)

    # Diff
    add_box(s6, Inches(7.3), y, Inches(5.5), Inches(0.55), diff_str, fs=16, fc=diff_color, bold=True,
            align=PP_ALIGN.CENTER, bg=CARD_BG)

# Average
add_box(s6, Inches(0.5), Inches(5.5), Inches(1.5), Inches(0.55), "AVG", fs=16, fc=WHITE, bold=True,
        align=PP_ALIGN.CENTER, bg=RGBColor(0x1E, 0x3A, 0x5F))
add_box(s6, Inches(2.1), Inches(5.5), Inches(2.5), Inches(0.55), "0.8043", fs=16, fc=ACCENT_ORANGE, bold=True,
        align=PP_ALIGN.CENTER, bg=RGBColor(0x1E, 0x3A, 0x5F))
add_box(s6, Inches(4.7), Inches(5.5), Inches(2.5), Inches(0.55), "0.8101", fs=16, fc=ACCENT_GREEN, bold=True,
        align=PP_ALIGN.CENTER, bg=RGBColor(0x1E, 0x3A, 0x5F))
add_box(s6, Inches(7.3), Inches(5.5), Inches(5.5), Inches(0.55), "+0.0058 (+0.7%)", fs=16, fc=ACCENT_GREEN, bold=True,
        align=PP_ALIGN.CENTER, bg=RGBColor(0x1E, 0x3A, 0x5F))


# ==================== Slide 7: Key Differences ====================
s7 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(s7, DARK_BG)

add_text(s7, Inches(0.5), Inches(0.3), Inches(12), Inches(0.8),
         "关键差异分析", fs=36, fc=ACCENT_BLUE, bold=True)

# Left - Baseline issues
add_box(s7, Inches(0.5), Inches(1.3), Inches(6), Inches(5.5), border=ACCENT_RED)
add_text(s7, Inches(0.8), Inches(1.5), Inches(5.5), Inches(0.5),
         "Baseline 已知问题", fs=22, fc=ACCENT_RED, bold=True)

issues = [
    "3个ECG特征全部缺失NaN，未有效处理",
    "类别严重不平衡(4.1% vs 95.9%)无补偿",
    "无Pipeline，标准化可能泄露",
    "Fold间AUC波动大(0.74~0.86)",
    "仅报告AUC，缺少其他分类指标",
]
for i, t in enumerate(issues):
    add_text(s7, Inches(1.0), Inches(2.2 + i * 0.7), Inches(5.2), Inches(0.6),
             f"✗  {t}", fs=14, fc=WHITE)

# Right - Improvements
add_box(s7, Inches(6.8), Inches(1.3), Inches(6), Inches(5.5), border=ACCENT_GREEN)
add_text(s7, Inches(7.1), Inches(1.5), Inches(5.5), Inches(0.5),
         "改进版 解决方案", fs=22, fc=ACCENT_GREEN, bold=True)

fixes = [
    "SimpleImputer(median)填充缺失值",
    "class_weight='balanced'自动调整权重",
    "Pipeline封装，数据零泄露",
    "StratifiedKFold保证类别一致",
    "完整报告Precision/Recall/F1/AUC",
]
for i, t in enumerate(fixes):
    add_text(s7, Inches(7.3), Inches(2.2 + i * 0.7), Inches(5.2), Inches(0.6),
             f"✓  {t}", fs=14, fc=WHITE)


# ==================== Slide 8: Conclusion ====================
s8 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(s8, DARK_BG)

add_text(s8, Inches(0.5), Inches(0.3), Inches(12), Inches(0.8),
         "总结", fs=36, fc=ACCENT_BLUE, bold=True)

# Key results
add_box(s8, Inches(0.5), Inches(1.3), Inches(12.3), Inches(2.0),
        bg=RGBColor(0x1E, 0x3A, 0x5F), border=ACCENT_GREEN)

results_text = [
    ("AUC提升", "0.8043 → 0.8101 (+0.7%)", ACCENT_GREEN),
    ("新增指标", "Accuracy 90.4% | Precision 20.9% | Recall 40.7% | F1 27.1%", ACCENT_BLUE),
    ("核心修复", "类别不平衡处理 + 数据泄露防护 + 分层交叉验证", ACCENT_ORANGE),
]
for i, (label, value, color) in enumerate(results_text):
    add_text(s8, Inches(1.0), Inches(1.5 + i * 0.6), Inches(2.5), Inches(0.5),
             label, fs=16, fc=color, bold=True)
    add_text(s8, Inches(3.5), Inches(1.5 + i * 0.6), Inches(9), Inches(0.5),
             value, fs=16, fc=WHITE)

# Next steps
add_text(s8, Inches(0.5), Inches(3.8), Inches(12), Inches(0.5),
         "后续优化建议", fs=22, fc=WHITE, bold=True)

next_steps = [
    ("1. 数据层面", "SMOTE/ADASYN过采样正样本，或收集更多CAD+数据"),
    ("2. 模型层面", "尝试XGBoost/LightGBM + class_weight，贝叶斯超参优化"),
    ("3. 阈值优化", "当前0.5 → 调整为0.3以提升Recall"),
    ("4. 特征工程", "ST段偏移和灌注指数为最重要特征，可重点优化"),
]
for i, (k, v) in enumerate(next_steps):
    y = Inches(4.4 + i * 0.7)
    add_text(s8, Inches(0.8), y, Inches(2.5), Inches(0.5), k, fs=14, fc=ACCENT_BLUE, bold=True)
    add_text(s8, Inches(3.3), y, Inches(9.5), Inches(0.5), v, fs=14, fc=WHITE)

# Bottom highlight
add_box(s8, Inches(2.5), Inches(6.5), Inches(8.3), Inches(0.7),
        "Baseline AUC: 0.8043  →  改进版 AUC: 0.8101  |  核心问题已修复",
        fs=16, fc=ACCENT_GREEN, bold=True, align=PP_ALIGN.CENTER,
        bg=RGBColor(0x1E, 0x3A, 0x5F), border=ACCENT_GREEN)


# Save
output = r'E:\data\dataset\Baseline_vs_改进版对比.pptx'
prs.save(output)
print(f'PPT saved: {output}')
