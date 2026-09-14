"""Generate CAD 5-fold CV report PPT (10 features including ECG-derived)."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import json, pandas as pd, numpy as np
from pathlib import Path

OUTPUT_DIR = Path(r"E:\data\attachments_cv_output")
PPT_PATH = OUTPUT_DIR / "CAD_5foldCV_10feat_report.pptx"

with open(OUTPUT_DIR / "cv_results.json") as f:
    res = json.load(f)

res["n"] = res["total_samples"]
res["pos"] = res["positive_samples"]
res["neg"] = res["negative_samples"]

df = pd.read_csv(OUTPUT_DIR / "extracted_features.csv")

BLUE = RGBColor(0, 51, 102)
DARK = RGBColor(40, 40, 40)
GRAY = RGBColor(100, 100, 100)
WHITE = RGBColor(255, 255, 255)
LIGHT_BG = RGBColor(240, 245, 250)
RED_BG = RGBColor(255, 235, 235)
GREEN_BG = RGBColor(235, 255, 235)

def _set_cell(cell, text, bold=False, bg=None, font_color=None, size=9):
    cell.text = str(text)
    for p in cell.text_frame.paragraphs:
        p.font.size = Pt(size)
        if bold: p.font.bold = True
        if font_color: p.font.color.rgb = font_color
    if bg:
        from pptx.oxml.ns import qn
        tcPr = cell._tc.get_or_add_tcPr()
        solidFill = tcPr.makeelement(qn('a:solidFill'), {})
        srgbClr = solidFill.makeelement(qn('a:srgbClr'), {
            'val': '%02x%02x%02x' % (
                bg[0] if isinstance(bg, tuple) else bg.red,
                bg[1] if isinstance(bg, tuple) else bg.green,
                bg[2] if isinstance(bg, tuple) else bg.blue)})
        solidFill.append(srgbClr)
        tcPr.append(solidFill)

def _add_text(slide, left, top, width, height, text, size=14, bold=False, color=None, align=None):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    if bold: p.font.bold = True
    if color: p.font.color.rgb = color
    if align: p.alignment = align
    return tf

FEATURES = ["resting_hr_bpm", "sdnn_ms", "rmssd_ms", "lf_hf_ratio", "perfusion_index",
            "augmentation_index", "primary_reflected_peak_delay_ms", "st_deviation_mv",
            "qtc_ms", "pulse_transit_time_ms"]

feat_stats = df[FEATURES].describe().T
feat_stats["median"] = df[FEATURES].median()

pos_df = df[df["label"] == 1]
neg_df = df[df["label"] == 0]

prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]

# ── Slide 1: Title ──
sl = prs.slides.add_slide(blank)
_add_text(sl, 0.8, 0.8, 11.5, 1.2, "CAD\u98ce\u9669\u9884\u6d4b\u6a21\u578b \u2014 \u4e94\u6298\u4ea4\u53c9\u9a8c\u8bc1\u62a5\u544a", 36, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
_add_text(sl, 0.8, 2.2, 11.5, 0.8, "\u4ec5\u4f7f\u7528Attachments\u6570\u636e | 10\u4e2a\u751f\u7406\u7279\u5f81\uff087 PPG + 3 ECG-derived\uff09", 22, color=GRAY, align=PP_ALIGN.CENTER)
_add_text(sl, 0.8, 3.2, 11.5, 0.8, "XGBoost + Stratified 5-Fold CV", 18, color=GRAY, align=PP_ALIGN.CENTER)
_add_text(sl, 0.8, 4.5, 11.5, 0.6, f"\u6837\u672c\u6570: {res['n']} | \u6b63\u6837\u672c: {res['pos']} ({res['pos']/res['n']*100:.1f}%) | \u8d1f\u6837\u672c: {res['neg']} ({res['neg']/res['n']*100:.1f}%)", 16, color=DARK, align=PP_ALIGN.CENTER)
_add_text(sl, 0.8, 5.5, 11.5, 0.6, "2026-09-10", 14, color=GRAY, align=PP_ALIGN.CENTER)

# ── Slide 2: Overview ──
sl2 = prs.slides.add_slide(blank)
_add_text(sl2, 0.5, 0.3, 12, 0.8, "\u4e00\u3001\u7814\u7a76\u6982\u8ff0", 28, bold=True, color=BLUE)

items = [
    ("\u6570\u636e\u6765\u6e90", f"{res['n']} \u4e2a\u6709\u6548zip\u6587\u4ef6\uff081000\u4e2a\u603b\u8ba1\uff09"),
    ("\u6807\u7b7e\u6620\u5c04", "\u901a\u8fc7\u6807\u7b7e\u5bf9\u5e94\u8868.txt \u5c06zip\u65f6\u95f4\u6233\u6620\u5c04\u5230\u5916\u90e8ID\uff0c\u57fa\u4e8e\u8bca\u65ad\u5173\u952e\u8bcd\u6807\u6ce8CAD"),
    ("\u7279\u5f81\u63d0\u53d6", "\u4ecePPG\u4fe1\u53f7\u63d0\u53d67\u4e2a\u751f\u7406\u7279\u5f81 + 3\u4e2aECG-derived\u7279\u5f81\uff08ST\u6bb5\u504f\u79fb\u3001QTc\u3001\u8109\u640f\u4f20\u5bfc\u65f6\u95f4\uff09"),
    ("\u6a21\u578b", "XGBoost (max_depth=4, n_estimators=300, lr=0.04)"),
    ("\u8bc4\u4f30\u65b9\u6cd5", "\u5206\u5c42\u4e94\u6298\u4ea4\u53c9\u9a8c\u8bc1 (Stratified 5-Fold CV, shuffle=True, seed=42)"),
    ("\u7f3a\u5931\u503c\u5904\u7406", "SimpleImputer (median + add_indicator)"),
]
for i, (k, v) in enumerate(items):
    _add_text(sl2, 0.8, 1.2 + i*0.8, 2.2, 0.5, k, 14, bold=True, color=DARK)
    _add_text(sl2, 3.1, 1.2 + i*0.8, 9.5, 0.6, v, 13, color=DARK)

# ── Slide 3: Results ──
sl3 = prs.slides.add_slide(blank)
_add_text(sl3, 0.5, 0.3, 12, 0.8, "\u4e8c\u3001\u4e94\u6298\u4ea4\u53c9\u9a8c\u8bc1\u7ed3\u679c", 28, bold=True, color=BLUE)

kpis = [
    ("Mean AUC", f"{res['mean_auc']:.4f}", BLUE),
    ("Std AUC", f"\u00b1{res['std_auc']:.4f}", GRAY),
    ("\u6837\u672c\u91cf", f"{res['n']}", DARK),
    ("CAD+\u9633\u6027", f"{res['pos']} ({res['pos']/res['n']*100:.1f}%)", RGBColor(180,0,0)),
    ("CAD-\u9634\u6027", f"{res['neg']} ({res['neg']/res['n']*100:.1f}%)", RGBColor(0,120,0)),
]
for i, (label, val, clr) in enumerate(kpis):
    x = 0.5 + i * 2.5
    shape = sl3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.2), Inches(2.2), Inches(1.2))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(245, 248, 252)
    shape.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = val
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = clr
    p.alignment = PP_ALIGN.CENTER
    p2 = tf.add_paragraph()
    p2.text = label
    p2.font.size = Pt(11)
    p2.font.color.rgb = GRAY
    p2.alignment = PP_ALIGN.CENTER

tbl = sl3.shapes.add_table(7, 3, Inches(1.5), Inches(2.8), Inches(5), Inches(3.2)).table
for j, h in enumerate(["Fold", "AUC", "\u6837\u672c\u6570"]):
    _set_cell(tbl.cell(0, j), h, bold=True, bg=BLUE, font_color=WHITE, size=11)
fold_n = [182, 182, 182, 182, 181]
for i, (auc, n) in enumerate(zip(res["fold_aucs"], fold_n)):
    bg = LIGHT_BG if i % 2 == 0 else None
    _set_cell(tbl.cell(i+1, 0), f"Fold {i+1}", bg=bg, size=10)
    _set_cell(tbl.cell(i+1, 1), f"{auc:.4f}", bg=bg, size=10)
    _set_cell(tbl.cell(i+1, 2), str(n), bg=bg, size=10)
_set_cell(tbl.cell(6, 0), "Mean", bold=True, bg=RGBColor(230,240,250), size=10)
_set_cell(tbl.cell(6, 1), f"{res['mean_auc']:.4f}", bold=True, bg=RGBColor(230,240,250), size=10)
_set_cell(tbl.cell(6, 2), str(res['n']), bold=True, bg=RGBColor(230,240,250), size=10)

_add_text(sl3, 7.5, 2.8, 5, 0.5, "AUC\u5206\u5e03", 16, bold=True, color=DARK)
for i, auc in enumerate(res["fold_aucs"]):
    y = 3.5 + i * 0.55
    bar_w = max((auc - 0.5) / 0.5 * 4.5, 0.05)
    shape = sl3.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.5), Inches(y), Inches(bar_w), Inches(0.35))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0, 102, 204) if auc >= res['mean_auc'] else RGBColor(180, 200, 230)
    shape.line.fill.background()
    _add_text(sl3, 7.5 + bar_w + 0.1, y, 1, 0.35, f"{auc:.4f}", 10, color=DARK)
    _add_text(sl3, 6.5, y, 0.9, 0.35, f"F{i+1}", 10, color=GRAY, align=PP_ALIGN.RIGHT)

# ── Slide 4: Feature Statistics (all 10) ──
sl4 = prs.slides.add_slide(blank)
_add_text(sl4, 0.5, 0.3, 12, 0.8, "\u4e09\u3001\u5168\u90e810\u4e2a\u7279\u5f81\u7edf\u8ba1", 28, bold=True, color=BLUE)

feat_names_cn = {
    "resting_hr_bpm": "\u9759\u606f\u5fc3\u7387(bpm)",
    "sdnn_ms": "SDNN(ms)",
    "rmssd_ms": "RMSSD(ms)",
    "lf_hf_ratio": "LF/HF\u6bd4\u503c",
    "perfusion_index": "\u704f\u6ce8\u6307\u6570",
    "augmentation_index": "\u589e\u5f3a\u6307\u6570",
    "primary_reflected_peak_delay_ms": "\u53cd\u5c04\u5cf0\u5ef6\u8fdf(ms)",
    "st_deviation_mv": "ST\u6bb5\u504f\u79fb(mV)",
    "qtc_ms": "QTc(ms)",
    "pulse_transit_time_ms": "\u8109\u640f\u4f20\u5bfc\u65f6\u95f4(ms)",
}
feat_cats = ["PPG"]*7 + ["ECG-derived"]*3

tbl2 = sl4.shapes.add_table(len(FEATURES)+1, 7, Inches(0.2), Inches(1.2), Inches(12.9), Inches(5.8)).table
for j, h in enumerate(["\u7279\u5f81", "\u4e2d\u6587", "\u5747\u503c", "\u4e2d\u4f4d\u6570", "\u6807\u51c6\u5dee", "\u6700\u5c03\u503c", "\u6700\u5927\u503c"]):
    _set_cell(tbl2.cell(0, j), h, bold=True, bg=BLUE, font_color=WHITE, size=10)
for i, feat in enumerate(FEATURES):
    bg = LIGHT_BG if i % 2 == 0 else None
    st = feat_stats.loc[feat]
    row_vals = [
        feat,
        feat_names_cn.get(feat, feat),
        f"{st['mean']:.2f}",
        f"{st['median']:.2f}",
        f"{st['std']:.2f}",
        f"{st['min']:.2f}",
        f"{st['max']:.2f}",
    ]
    for j, val in enumerate(row_vals):
        cell_bg = bg
        if feat_cats[i] == "ECG-derived" and j == 0:
            cell_bg = RGBColor(255, 245, 230)
        _set_cell(tbl2.cell(i+1, j), val, bold=(j==0), bg=cell_bg, size=9)

# ── Slide 5: CAD+ vs CAD- ──
sl5 = prs.slides.add_slide(blank)
_add_text(sl5, 0.5, 0.3, 12, 0.8, "\u56db\u3001CAD\u9633\u6027 vs \u9634\u6027 \u7279\u5f81\u5bf9\u6bd4", 28, bold=True, color=BLUE)

tbl3 = sl5.shapes.add_table(len(FEATURES)+1, 4, Inches(0.5), Inches(1.2), Inches(12), Inches(5.8)).table
for j, h in enumerate(["\u7279\u5f81", f"CAD+ (n={len(pos_df)})", f"CAD- (n={len(neg_df)})", "\u5dee\u5f02"]):
    _set_cell(tbl3.cell(0, j), h, bold=True, bg=BLUE, font_color=WHITE, size=10)
for i, feat in enumerate(FEATURES):
    bg = LIGHT_BG if i % 2 == 0 else None
    pos_mean = pos_df[feat].mean()
    neg_mean = neg_df[feat].mean()
    diff = pos_mean - neg_mean
    _set_cell(tbl3.cell(i+1, 0), feat_names_cn.get(feat, feat), bold=True, bg=bg, size=10)
    _set_cell(tbl3.cell(i+1, 1), f"{pos_mean:.2f}", bg=bg, size=10)
    _set_cell(tbl3.cell(i+1, 2), f"{neg_mean:.2f}", bg=bg, size=10)
    diff_str = f"{diff:+.2f}"
    _set_cell(tbl3.cell(i+1, 3), diff_str, bg=RED_BG if diff > 0 else GREEN_BG if diff < 0 else bg, size=10)

_add_text(sl5, 0.5, 6.8, 12, 0.5,
    "\u6ce8: \u6837\u672c\u91cf\u6781\u4e0d\u5e73\u8861(CAD+\u4ec5" + str(len(pos_df)) + "\u4f8b)\uff0c\u5bf9\u6bd4\u4ec5\u4f9b\u53c2\u8003", 11, color=GRAY)

# ── Slide 6: Feature descriptions ──
sl6 = prs.slides.add_slide(blank)
_add_text(sl6, 0.5, 0.3, 12, 0.8, "\u4e94\u3001\u8f93\u5165\u7279\u5f81\u8bf4\u660e", 28, bold=True, color=BLUE)

feat_desc = [
    ("resting_hr_bpm", "\u9759\u606f\u5fc3\u7387", "PPG\u8109\u640f\u95f4\u9694\u4e2d\u4f4d\u6570", "PPG"),
    ("sdnn_ms", "SDNN", "RR\u95f4\u671f\u6807\u51c6\u5dee\uff0c\u81ea\u4e3b\u795e\u7ecf\u6574\u4f53\u8c03\u8282", "HRV\u65f6\u57df"),
    ("rmssd_ms", "RMSSD", "\u76f8\u90bbRR\u95f4\u671f\u5dee\u503c\u5747\u65b9\u6839\uff0c\u526f\u4ea4\u611f\u795e\u7ecf\u6d3b\u6027", "HRV\u65f6\u57df"),
    ("lf_hf_ratio", "LF/HF\u6bd4\u503c", "\u4f4e\u9891/\u9ad8\u9891\u529f\u7387\u6bd4\uff0c\u4ea4\u611f-\u526f\u4ea4\u611f\u5e73\u8861", "HRV\u9891\u57df"),
    ("perfusion_index", "\u704f\u6ce8\u6307\u6570", "PPG\u4fe1\u53f7AC/DC\u632f\u5e45\u767e\u5206\u6bd4", "\u8840\u7ba1\u529f\u80fd"),
    ("augmentation_index", "\u589e\u5f3a\u6307\u6570", "\u53cd\u5c04\u6ce2\u4e0e\u5165\u5c04\u6ce2\u5e45\u503c\u6bd4", "\u52a8\u8109\u529f\u80fd"),
    ("primary_reflected_peak_delay_ms", "\u53cd\u5c04\u5cf0\u5ef6\u8fdf", "\u5165\u5c04\u6ce2\u5230\u53cd\u5c04\u6ce2\u7684\u65f6\u95f4\u5dee", "\u8840\u7ba1\u529f\u80fd"),
    ("st_deviation_mv", "ST\u6bb5\u504f\u79fb", "PPG ST\u6bb5\u5747\u503c\u4e0e\u57fa\u7ebf\u504f\u5dee\uff08\u8fd1\u4f3cECG\uff09", "ECG-derived"),
    ("qtc_ms", "QTc\u95f4\u671f", "\u6821\u6b63QT\u95f4\u671f\uff0cRR\u95f4\u671f\u5f02\u5e38\u7b5b\u67e5", "ECG-derived"),
    ("pulse_transit_time_ms", "\u8109\u640f\u4f20\u5bfc\u65f6\u95f4", "PPG\u5cf0\u95f4\u9694\u5747\u503c\uff0c\u8fd1\u4f3c\u8109\u640f\u4f20\u5bfc\u65f6\u95f4", "ECG-derived"),
]
tbl4 = sl6.shapes.add_table(len(feat_desc)+1, 4, Inches(0.3), Inches(1.2), Inches(12.7), Inches(5.8)).table
for j, h in enumerate(["\u7279\u5f81\u540d", "\u4e2d\u6587", "\u8bf4\u660e", "\u7c7b\u522b"]):
    _set_cell(tbl4.cell(0, j), h, bold=True, bg=BLUE, font_color=WHITE, size=10)
for i, (f, cn, desc, cat) in enumerate(feat_desc):
    bg = LIGHT_BG if i % 2 == 0 else None
    _set_cell(tbl4.cell(i+1, 0), f, bold=True, bg=RGBColor(255,245,230) if cat == "ECG-derived" else bg, size=9)
    _set_cell(tbl4.cell(i+1, 1), cn, bg=bg, size=9)
    _set_cell(tbl4.cell(i+1, 2), desc, bg=bg, size=9)
    _set_cell(tbl4.cell(i+1, 3), cat, bg=RGBColor(255,245,230) if cat == "ECG-derived" else bg, size=9)

# ── Slide 7: Conclusions ──
sl7 = prs.slides.add_slide(blank)
_add_text(sl7, 0.5, 0.3, 12, 0.8, "\u516d\u3001\u7ed3\u8bba\u4e0e\u5efa\u8bae", 28, bold=True, color=BLUE)

conclusions = [
    f"1. \u57fa\u7ebfAUC = {res['mean_auc']:.4f} (\u00b1{res['std_auc']:.4f})\uff0c\u4f7f\u7528\u5168\u90e810\u4e2a\u7279\u5f81\uff08\u542b3\u4e2aECG-derived\uff09",
    f"2. \u6570\u636e\u4e25\u91cd\u4e0d\u5e73\u8861: CAD\u9633\u6027\u4ec5{res['pos']}\u4f8b({res['pos']/res['n']*100:.1f}%)\uff0c\u9700\u5173\u6ce8\u5047\u9634\u6027\u98ce\u9669",
    "3. 3\u4e2aECG-derived\u7279\u5f81\u5df2\u52a0\u5165\u9a8c\u8bc1\uff1aST\u6bb5\u504f\u79fb\u3001QTc\u95f4\u671f\u3001\u8109\u640f\u4f20\u5bfc\u65f6\u95f4",
    f"4. Fold\u95f4AUC\u6ce2\u52a8\u8f83\u5c0f({min(res['fold_aucs']):.4f}~{max(res['fold_aucs']):.4f})\uff0c\u7a33\u5b9a\u6027\u826f\u597d",
    "",
    "\u6539\u8fdb\u5efa\u8bae:",
    "  a) \u6536\u96c6\u66f4\u591aCAD\u9633\u6027\u6837\u672c\uff0c\u6539\u5584\u7c7b\u522b\u5e73\u8861",
    "  b) \u5f15\u5165\u5b9e\u9645ECG\u4fe1\u53f7\u91c7\u96c6\uff0c\u66ff\u4ee3PPG\u8fd1\u4f3c\u8ba1\u7b97",
    "  c) \u5c1d\u8bd5SMOTE\u8fc7\u91c7\u6837\u6216\u4ee3\u4ef7\u654f\u611f\u5b66\u4e60\u7f13\u89e3\u4e0d\u5e73\u8861",
    "  d) \u8c03\u6574\u5206\u7c7b\u9608\u503c(\u5f53\u524d0.5)\uff0c\u6839\u636e\u4e34\u5e8a\u9700\u6c42\u6743\u8861\u654f\u611f\u6027/\u7279\u5f02\u6027",
    "  e) \u5f15\u5165\u66f4\u591a\u4e34\u5e8a\u6307\u6807: \u5e74\u9f84\u3001\u6027\u522b\u3001\u8840\u538b\u7b49",
]
for i, line in enumerate(conclusions):
    clr = DARK if not line.startswith("  ") else GRAY
    bld = not line.startswith("  ") and line != ""
    _add_text(sl7, 0.8, 1.2 + i*0.5, 11.5, 0.45, line, 13, bold=bld, color=clr)

# ── Slide 8: Model Config ──
sl8 = prs.slides.add_slide(blank)
_add_text(sl8, 0.5, 0.3, 12, 0.8, "\u4e03\u3001\u6a21\u578b\u914d\u7f6e", 28, bold=True, color=BLUE)

config_items = [
    ("\u6a21\u578b\u7c7b\u578b", "XGBClassifier (XGBoost)"),
    ("max_depth", "4"),
    ("n_estimators", "300"),
    ("learning_rate", "0.04"),
    ("subsample", "0.8"),
    ("colsample_bytree", "0.8"),
    ("eval_metric", "logloss"),
    ("\u7279\u5f81\u6570", f"{len(FEATURES)}\u4e2a\uff087 PPG + 3 ECG-derived\uff09"),
    ("\u7f3a\u5931\u503c\u5904\u7406", "SimpleImputer (median + indicator)"),
    ("\u4ea4\u53c9\u9a8c\u8bc1", "StratifiedKFold, n_splits=5, shuffle=True, random_state=42"),
]
for i, (k, v) in enumerate(config_items):
    _add_text(sl8, 0.8, 1.2 + i*0.5, 3, 0.4, k, 13, bold=True, color=DARK)
    _add_text(sl8, 3.8, 1.2 + i*0.5, 8, 0.4, v, 13, color=DARK)

prs.save(str(PPT_PATH))
print(f"PPT saved to {PPT_PATH}")
