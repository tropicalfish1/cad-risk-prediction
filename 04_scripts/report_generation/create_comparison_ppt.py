"""生成五折交叉验证改进对比PPT"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Colors
DARK_BG = RGBColor(0x1A, 0x1A, 0x2E)
ACCENT_BLUE = RGBColor(0x00, 0x96, 0xD6)
ACCENT_GREEN = RGBColor(0x2E, 0xCC, 0x71)
ACCENT_RED = RGBColor(0xE7, 0x4C, 0x3C)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xCC, 0xCC, 0xCC)
CARD_BG = RGBColor(0x2D, 0x2D, 0x44)


def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_shape_with_text(slide, left, top, width, height, text, font_size=18,
                        font_color=WHITE, bold=False, alignment=PP_ALIGN.LEFT,
                        bg_color=None, border_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color or CARD_BG
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(2)
    else:
        shape.line.fill.background()

    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = font_color
    p.font.bold = bold
    p.alignment = alignment
    return shape


def add_textbox(slide, left, top, width, height, text, font_size=18,
                font_color=WHITE, bold=False, alignment=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = font_color
    p.font.bold = bold
    p.alignment = alignment
    return txBox


# ==================== Slide 1: Title ====================
slide1 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide1, DARK_BG)

add_textbox(slide1, Inches(1), Inches(1.5), Inches(11), Inches(1.2),
            "PPG信号五折交叉验证", font_size=44, font_color=ACCENT_BLUE, bold=True,
            alignment=PP_ALIGN.CENTER)

add_textbox(slide1, Inches(1), Inches(3.0), Inches(11), Inches(0.8),
            "改进方案对比分析", font_size=32, font_color=WHITE, bold=False,
            alignment=PP_ALIGN.CENTER)

add_shape_with_text(slide1, Inches(4.5), Inches(4.5), Inches(4), Inches(0.6),
                    "基于909例样本 | 10个PPG特征 | RandomForest",
                    font_size=14, font_color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER,
                    bg_color=DARK_BG)

add_textbox(slide1, Inches(1), Inches(6.0), Inches(11), Inches(0.5),
            "2026年9月", font_size=16, font_color=LIGHT_GRAY,
            alignment=PP_ALIGN.CENTER)


# ==================== Slide 2: Improvement Points ====================
slide2 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide2, DARK_BG)

add_textbox(slide2, Inches(0.5), Inches(0.3), Inches(12), Inches(0.8),
            "改进要点", font_size=36, font_color=ACCENT_BLUE, bold=True)

improvements = [
    ("1", "KFold → StratifiedKFold", "保证每折类别比例一致，避免验证集单类别问题"),
    ("2", "手动标准化 → Pipeline", "标准化仅在训练集fit，防止数据泄露到验证集"),
    ("3", "添加 class_weight='balanced'", "自动调整类别权重，缓解正负样本不平衡(1:23.6)"),
    ("4", "新增评估指标", "补充 Precision / Recall / F1，全面评估模型性能"),
    ("5", "处理全量数据", "从10个文件扩展到909例样本，结果更具统计意义"),
]

for i, (num, title, desc) in enumerate(improvements):
    y = Inches(1.3 + i * 1.15)
    # Number circle
    add_shape_with_text(slide2, Inches(0.7), y, Inches(0.6), Inches(0.6),
                        num, font_size=20, font_color=WHITE, bold=True,
                        alignment=PP_ALIGN.CENTER, bg_color=ACCENT_BLUE)
    # Title
    add_textbox(slide2, Inches(1.5), y - Inches(0.05), Inches(5), Inches(0.45),
                title, font_size=20, font_color=WHITE, bold=True)
    # Description
    add_textbox(slide2, Inches(1.5), y + Inches(0.35), Inches(10), Inches(0.45),
                desc, font_size=14, font_color=LIGHT_GRAY)


# ==================== Slide 3: Results Comparison Table ====================
slide3 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide3, DARK_BG)

add_textbox(slide3, Inches(0.5), Inches(0.3), Inches(12), Inches(0.8),
            "五折交叉验证结果对比", font_size=36, font_color=ACCENT_BLUE, bold=True)

# Table header
headers = ["指标", "原版 (KFold)", "改进版 (StratifiedKFold)", "提升"]
col_widths = [Inches(2.5), Inches(3), Inches(3.5), Inches(2.5)]
col_starts = [Inches(0.8)]
for w in col_widths[:-1]:
    col_starts.append(col_starts[-1] + w)

header_y = Inches(1.3)
for i, (header, x, w) in enumerate(zip(headers, col_starts, col_widths)):
    add_shape_with_text(slide3, x, header_y, w, Inches(0.55),
                        header, font_size=16, font_color=WHITE, bold=True,
                        alignment=PP_ALIGN.CENTER, bg_color=ACCENT_BLUE)

# Table rows
rows = [
    ("Accuracy", "N/A*", "0.9043 ± 0.028", "-"),
    ("AUC", "N/A*", "0.8101 ± 0.043", "-"),
    ("Precision", "未报告", "0.2085 ± 0.097", "新增"),
    ("Recall", "未报告", "0.4071 ± 0.094", "新增"),
    ("F1 Score", "未报告", "0.2705 ± 0.100", "新增"),
    ("样本数", "10个文件", "909例", "+8990%"),
    ("类别平衡", "可能失衡", "分层保证", "修复"),
]

for row_idx, (metric, old, new, change) in enumerate(rows):
    y = Inches(1.95 + row_idx * 0.65)
    bg = CARD_BG if row_idx % 2 == 0 else RGBColor(0x35, 0x35, 0x50)

    values = [metric, old, new, change]
    colors = [WHITE, LIGHT_GRAY, ACCENT_GREEN, ACCENT_BLUE]

    for col_idx, (val, color, x, w) in enumerate(zip(values, colors, col_starts, col_widths)):
        add_shape_with_text(slide3, x, y, w, Inches(0.55),
                            val, font_size=14, font_color=color,
                            alignment=PP_ALIGN.CENTER, bg_color=bg)

add_textbox(slide3, Inches(0.8), Inches(6.5), Inches(11), Inches(0.5),
            "* 原版仅处理10个文件，数据量不足以产生可靠的统计指标",
            font_size=12, font_color=LIGHT_GRAY)


# ==================== Slide 4: Fold-by-Fold Results ====================
slide4 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide4, DARK_BG)

add_textbox(slide4, Inches(0.5), Inches(0.3), Inches(12), Inches(0.8),
            "各折详细结果", font_size=36, font_color=ACCENT_BLUE, bold=True)

fold_headers = ["Fold", "Accuracy", "AUC", "Precision", "Recall", "F1", "正样本/验证集"]
fold_col_widths = [Inches(1), Inches(1.6), Inches(1.6), Inches(1.8), Inches(1.6), Inches(1.4), Inches(2.2)]
fold_col_starts = [Inches(0.7)]
for w in fold_col_widths[:-1]:
    fold_col_starts.append(fold_col_starts[-1] + w)

header_y = Inches(1.2)
for header, x, w in zip(fold_headers, fold_col_starts, fold_col_widths):
    add_shape_with_text(slide4, x, header_y, w, Inches(0.5),
                        header, font_size=14, font_color=WHITE, bold=True,
                        alignment=PP_ALIGN.CENTER, bg_color=ACCENT_BLUE)

fold_data = [
    ("1", "0.8736", "0.7355", "0.1364", "0.4286", "0.2069", "7/182"),
    ("2", "0.9505", "0.8653", "0.4000", "0.5714", "0.4706", "7/182"),
    ("3", "0.8846", "0.8240", "0.1579", "0.3750", "0.2222", "8/182"),
    ("4", "0.8901", "0.7996", "0.1667", "0.3750", "0.2308", "8/182"),
    ("5", "0.9227", "0.8259", "0.1818", "0.2857", "0.2222", "7/181"),
]

for row_idx, row in enumerate(fold_data):
    y = Inches(1.85 + row_idx * 0.6)
    bg = CARD_BG if row_idx % 2 == 0 else RGBColor(0x35, 0x35, 0x50)
    for col_idx, (val, x, w) in enumerate(zip(row, fold_col_starts, fold_col_widths)):
        color = WHITE if col_idx == 0 else LIGHT_GRAY
        add_shape_with_text(slide4, x, y, w, Inches(0.5),
                            val, font_size=13, font_color=color,
                            alignment=PP_ALIGN.CENTER, bg_color=bg)

# Average row
avg_y = Inches(1.85 + 5 * 0.6)
avg_data = ("AVG", "0.9043", "0.8101", "0.2085", "0.4071", "0.2705", "37/909")
for col_idx, (val, x, w) in enumerate(zip(avg_data, fold_col_starts, fold_col_widths)):
    color = ACCENT_GREEN if col_idx > 0 else WHITE
    add_shape_with_text(slide4, x, avg_y, w, Inches(0.5),
                        val, font_size=14, font_color=color, bold=True,
                        alignment=PP_ALIGN.CENTER, bg_color=RGBColor(0x1E, 0x3A, 0x5F))


# ==================== Slide 5: Feature Importance ====================
slide5 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide5, DARK_BG)

add_textbox(slide5, Inches(0.5), Inches(0.3), Inches(12), Inches(0.8),
            "特征重要性排名", font_size=36, font_color=ACCENT_BLUE, bold=True)

features = [
    ("ST段偏移 (st_deviation_mv)", 0.196, "反映心肌缺血状态"),
    ("灌注指数 (perfusion_index)", 0.151, "反映末梢血液循环"),
    ("RMSSD (rmssd_ms)", 0.138, "反映副交感神经活性"),
    ("SDNN (sdnn_ms)", 0.099, "反映自主神经整体变异"),
    ("反射峰延迟 (primary_reflected_peak_delay_ms)", 0.091, "反映动脉硬化程度"),
    ("增强指数 (augmentation_index)", 0.075, "反映动脉僵硬度"),
    ("静息心率 (resting_hr_bpm)", 0.074, "基础心血管指标"),
    ("校正QT间期 (qtc_ms)", 0.063, "反映心室复极时间"),
    ("脉搏传导时间 (pulse_transit_time_ms)", 0.062, "反映血压变化"),
    ("LF/HF比 (lf_hf_ratio)", 0.050, "反映自主神经平衡"),
]

max_importance = max(f[1] for f in features)

for i, (name, importance, desc) in enumerate(features):
    y = Inches(1.2 + i * 0.58)

    # Feature name
    add_textbox(slide5, Inches(0.5), y, Inches(4.5), Inches(0.35),
                name, font_size=13, font_color=WHITE, bold=True)

    # Bar background
    bar_left = Inches(5.2)
    bar_width = Inches(5.5)
    add_shape_with_text(slide5, bar_left, y + Inches(0.02), bar_width, Inches(0.3),
                        "", font_size=10, bg_color=RGBColor(0x35, 0x35, 0x50))

    # Bar fill
    fill_width = int(bar_width * importance / max_importance)
    if fill_width > 0:
        add_shape_with_text(slide5, bar_left, y + Inches(0.02), fill_width, Inches(0.3),
                            "", font_size=10, bg_color=ACCENT_BLUE)

    # Importance value
    add_textbox(slide5, Inches(10.8), y, Inches(1.5), Inches(0.35),
                f"{importance:.3f}", font_size=13, font_color=ACCENT_GREEN, bold=True,
                alignment=PP_ALIGN.RIGHT)

    # Description
    add_textbox(slide5, Inches(5.2), y + Inches(0.28), Inches(7), Inches(0.28),
                desc, font_size=10, font_color=LIGHT_GRAY)


# ==================== Slide 6: Conclusion ====================
slide6 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide6, DARK_BG)

add_textbox(slide6, Inches(0.5), Inches(0.3), Inches(12), Inches(0.8),
            "总结与建议", font_size=36, font_color=ACCENT_BLUE, bold=True)

# Left column - conclusions
add_shape_with_text(slide6, Inches(0.5), Inches(1.3), Inches(6), Inches(5.5),
                    "", bg_color=CARD_BG, border_color=ACCENT_BLUE)

add_textbox(slide6, Inches(0.8), Inches(1.5), Inches(5.5), Inches(0.5),
            "核心改进", font_size=22, font_color=ACCENT_BLUE, bold=True)

conclusions = [
    "修复了KFold导致的类别分布不一致问题",
    "通过Pipeline消除了数据泄露风险",
    "class_weight处理了1:23.6的极端不平衡",
    "新增Precision/Recall/F1全面评估",
    "数据量从10例扩展到909例",
]

for i, text in enumerate(conclusions):
    add_textbox(slide6, Inches(1.0), Inches(2.2 + i * 0.7), Inches(5.2), Inches(0.6),
                f"✓  {text}", font_size=14, font_color=WHITE)

# Right column - next steps
add_shape_with_text(slide6, Inches(6.8), Inches(1.3), Inches(6), Inches(5.5),
                    "", bg_color=CARD_BG, border_color=ACCENT_GREEN)

add_textbox(slide6, Inches(7.1), Inches(1.5), Inches(5.5), Inches(0.5),
            "后续优化方向", font_size=22, font_color=ACCENT_GREEN, bold=True)

next_steps = [
    "SMOTE / ADASYN 过采样正样本",
    "调整分类阈值(当前0.5→0.3)",
    "尝试XGBoost / LightGBM",
    "增加正样本数据采集",
    "贝叶斯超参数优化",
]

for i, text in enumerate(next_steps):
    add_textbox(slide6, Inches(7.3), Inches(2.2 + i * 0.7), Inches(5.2), Inches(0.6),
                f"→  {text}", font_size=14, font_color=WHITE)

# Key metric highlight
add_shape_with_text(slide6, Inches(3), Inches(6.2), Inches(7), Inches(0.8),
                    "AUC = 0.8101  |  Accuracy = 0.9043  |  样本 = 909例",
                    font_size=18, font_color=ACCENT_GREEN, bold=True,
                    alignment=PP_ALIGN.CENTER, bg_color=RGBColor(0x1E, 0x3A, 0x5F),
                    border_color=ACCENT_GREEN)


# Save
output_path = r'E:\data\dataset\交叉验证改进对比.pptx'
prs.save(output_path)
print(f'PPT saved to: {output_path}')
