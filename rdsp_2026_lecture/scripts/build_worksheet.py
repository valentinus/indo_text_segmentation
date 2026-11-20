"""Build the in-class annotation worksheet (Excel / Google Sheets compatible).

Sheets
  1_Codebook   label definitions + rules
  2_Annotate   20 comments, 3 annotators x (topic, sentiment), dropdowns
  3_Agreement  live formulas: % full agreement, Fleiss' kappa, majority label
  4_AI_labels  LLM labels (keep hidden until the reveal) + human-vs-AI match

Run:  python scripts/build_worksheet.py   (from rdsp_2026_lecture/)
"""
import os
import sys

from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))
from comments import ANNOTATION_SET, ROWS, SENTIMENTS, TOPICS  # noqa: E402

rows = {r[0]: r for r in ROWS}
items = [rows[i] for i in ANNOTATION_SET]
N = len(items)
FIRST, LAST = 2, N + 1
codes = list(TOPICS.keys())
SENT_CODES = ["POS", "NEU", "NEG"]
SENT_MAP = {"Positive": "POS", "Neutral": "NEU", "Negative": "NEG"}

HEAD = Font(bold=True, color="FFFFFF")
HEAD_FILL = PatternFill("solid", fgColor="9B2C1F")
INPUT_FILL = PatternFill("solid", fgColor="FFF4E5")
WRAP = Alignment(wrap_text=True, vertical="top")


def header(ws, labels, widths):
    for c, (lab, w) in enumerate(zip(labels, widths), start=1):
        cell = ws.cell(row=1, column=c, value=lab)
        cell.font, cell.fill, cell.alignment = HEAD, HEAD_FILL, Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.freeze_panes = "A2"


wb = Workbook()

# ---------------------------------------------------------------- 1 Codebook
cb = wb.active
cb.title = "1_Codebook"
header(cb, ["Code", "Topic label", "Use when the comment is mainly about...", "Example"],
       [8, 30, 60, 60])
desc = {
    "SCN": ("the look, sound or feel of the place (views, gates, light, nature, mood)", "The rows of gates feel like a tunnel of fire."),
    "CRW": ("number of visitors, tourist behaviour, impact on locals, tourism policy", "Five years ago it wasn't this crowded."),
    "CUL": ("religion, history, symbols, rituals, etiquette and manners", "Foxes are messengers of Inari."),
    "PRC": ("planning: transport, timing, opening hours, cost, route, what to bring", "Wear comfortable shoes, the steps get slippery."),
    "FDS": ("food, drinks, souvenirs, shops and their prices", "Matcha soft serve near the station 10/10."),
    "VID": ("the video itself or the creator (editing, music, camera, content)", "Music in this video is too loud."),
    "OTH": ("spam, self-promotion, personal stories with no place content, off-topic", "Check out my channel!!!"),
}
for i, c in enumerate(codes, start=2):
    for j, v in enumerate([c, TOPICS[c], desc[c][0], desc[c][1]], start=1):
        cb.cell(row=i, column=j, value=v).alignment = WRAP
r = len(codes) + 3
rules = [
    "RULES",
    "1. Work ALONE. Do not discuss labels with your teammates until everyone is finished.",
    "2. Give each comment exactly ONE topic: the main thing it talks about. If two topics fit, pick the one that dominates.",
    "3. Sentiment = the writer's attitude: POS (praise, joy), NEU (facts, questions, tips, mixed), NEG (complaint, disappointment, sarcasm).",
    "4. Sarcasm counts by its intended meaning ('Great, another...' = NEG).",
    "5. Read the English gloss for Japanese / Indonesian comments.",
    "6. Time limit: 10 minutes for 20 comments. Gut feeling is fine.",
]
for k, t in enumerate(rules):
    c = cb.cell(row=r + k, column=1, value=t)
    c.font = Font(bold=(k == 0))
    cb.merge_cells(start_row=r + k, start_column=1, end_row=r + k, end_column=4)

# ---------------------------------------------------------------- 2 Annotate
an = wb.create_sheet("2_Annotate")
header(an, ["#", "Comment", "English gloss (JA / ID)",
            "Topic A1", "Topic A2", "Topic A3", "Sent. A1", "Sent. A2", "Sent. A3"],
       [5, 60, 40, 10, 10, 10, 10, 10, 10])
for k, it in enumerate(items):
    row = FIRST + k
    an.cell(row=row, column=1, value=k + 1)
    an.cell(row=row, column=2, value=it[1]).alignment = WRAP
    an.cell(row=row, column=3, value=it[3]).alignment = WRAP
    for col in range(4, 10):
        an.cell(row=row, column=col).fill = INPUT_FILL
    an.row_dimensions[row].height = 48
dv_t = DataValidation(type="list", formula1='"' + ",".join(codes) + '"', allow_blank=True)
dv_s = DataValidation(type="list", formula1='"' + ",".join(SENT_CODES) + '"', allow_blank=True)
an.add_data_validation(dv_t)
an.add_data_validation(dv_s)
dv_t.add(f"D{FIRST}:F{LAST}")
dv_s.add(f"G{FIRST}:I{LAST}")
an.cell(row=LAST + 2, column=2,
        value="A1 / A2 / A3 = annotator 1, 2, 3. Choose from the dropdown (or type the code).").font = Font(italic=True)

# ---------------------------------------------------------------- 3 Agreement


def agreement_block(ws, col0, title, src_cols, cats):
    """Per-item counts per category, P_i, full agreement and majority label.
    col0: first column index for this block. Returns dict of key cell refs."""
    a, b, c = src_cols
    ncat = len(cats)
    labels = [f"{title}: all 3 agree"] + [f"n({x})" for x in cats] + [f"{title}: P_i", f"{title}: majority"]
    for j, lab in enumerate(labels):
        cell = ws.cell(row=1, column=col0 + j, value=lab)
        cell.font, cell.fill, cell.alignment = HEAD, HEAD_FILL, Alignment(wrap_text=True)
        ws.column_dimensions[get_column_letter(col0 + j)].width = 9 if j else 12
    cnt_first = get_column_letter(col0 + 1)
    cnt_last = get_column_letter(col0 + ncat)
    for k in range(N):
        row = FIRST + k
        src = f"'2_Annotate'!{a}{row}:{c}{row}"
        full = f'=IF(COUNTA({src})<3,"",IF(AND(\'2_Annotate\'!{a}{row}=\'2_Annotate\'!{b}{row},\'2_Annotate\'!{b}{row}=\'2_Annotate\'!{c}{row}),1,0))'
        ws.cell(row=row, column=col0, value=full)
        for j, cat in enumerate(cats):
            ws.cell(row=row, column=col0 + 1 + j, value=f'=COUNTIF({src},"{cat}")')
        rng = f"{cnt_first}{row}:{cnt_last}{row}"
        ws.cell(row=row, column=col0 + 1 + ncat,
                value=f'=IF(SUM({rng})<3,"",(SUMSQ({rng})-3)/6)')
        hdr = f"${cnt_first}$1:${cnt_last}$1"
        ws.cell(row=row, column=col0 + 2 + ncat,
                value=f'=IF(SUM({rng})<3,"",IF(MAX({rng})>=2,SUBSTITUTE(SUBSTITUTE(INDEX({hdr},1,MATCH(MAX({rng}),{rng},0)),"n(",""),")",""),"none"))')
    return {
        "full": f"{get_column_letter(col0)}{FIRST}:{get_column_letter(col0)}{LAST}",
        "counts_cols": [get_column_letter(col0 + 1 + j) for j in range(ncat)],
        "pi": f"{get_column_letter(col0 + 1 + ncat)}{FIRST}:{get_column_letter(col0 + 1 + ncat)}{LAST}",
        "majority_col": get_column_letter(col0 + 2 + ncat),
    }


ag = wb.create_sheet("3_Agreement")
ag.cell(row=1, column=1, value="#").font = HEAD
ag.cell(row=1, column=1).fill = HEAD_FILL
ag.column_dimensions["A"].width = 5
for k in range(N):
    ag.cell(row=FIRST + k, column=1, value=k + 1)
T = agreement_block(ag, 2, "Topic", ("D", "E", "F"), codes)
S = agreement_block(ag, 2 + len(codes) + 3 + 1, "Sentiment", ("G", "H", "I"), SENT_CODES)

# summary panel (below the table)
sr = LAST + 3
ag.cell(row=sr, column=2, value="SUMMARY (fills in automatically)").font = Font(bold=True, size=13)


def fleiss(prefix_row, name, blk, ncats):
    r0 = prefix_row
    ag.cell(row=r0, column=2, value=name).font = Font(bold=True)
    ag.cell(row=r0 + 1, column=2, value="Items fully labelled")
    ag.cell(row=r0 + 1, column=5, value=f'=COUNT({blk["full"]})')
    ag.cell(row=r0 + 2, column=2, value="% all 3 annotators agree")
    ag.cell(row=r0 + 2, column=5, value=f'=IF(E{r0+1}=0,"",AVERAGE({blk["full"]}))')
    ag.cell(row=r0 + 2, column=5).number_format = "0%"
    ag.cell(row=r0 + 3, column=2, value="Observed agreement  P-bar")
    ag.cell(row=r0 + 3, column=5, value=f'=IF(E{r0+1}=0,"",AVERAGE({blk["pi"]}))')
    # expected agreement: sum over categories of p_j^2
    pj = "+".join(
        f"(SUM({c}{FIRST}:{c}{LAST})/(3*E{r0+1}))^2" for c in blk["counts_cols"])
    ag.cell(row=r0 + 4, column=2, value="Chance agreement  P-e")
    ag.cell(row=r0 + 4, column=5, value=f'=IF(E{r0+1}=0,"",{pj})')
    ag.cell(row=r0 + 5, column=2, value="Fleiss' kappa")
    ag.cell(row=r0 + 5, column=5, value=f'=IF(E{r0+1}=0,"",(E{r0+3}-E{r0+4})/(1-E{r0+4}))')
    ag.cell(row=r0 + 5, column=5).font = Font(bold=True, size=14, color="9B2C1F")
    ag.cell(row=r0 + 6, column=2, value="Interpretation (Landis & Koch)")
    k = f"E{r0+5}"
    ag.cell(row=r0 + 6, column=5,
            value=f'=IF({k}="","",IF({k}<0,"poor",IF({k}<=0.2,"slight",IF({k}<=0.4,"fair",IF({k}<=0.6,"moderate",IF({k}<=0.8,"substantial","almost perfect"))))))')
    for rr in range(r0 + 2, r0 + 6):
        ag.cell(row=rr, column=5).number_format = "0.000" if rr != r0 + 2 else "0%"
    return r0 + 8


nxt = fleiss(sr + 1, "TOPIC (7 labels)", T, len(codes))
fleiss(nxt, "SENTIMENT (3 labels)", S, 3)
ag.freeze_panes = "B2"

# ---------------------------------------------------------------- 4 AI labels
ai = wb.create_sheet("4_AI_labels")
header(ai, ["#", "Comment", "AI topic", "AI secondary", "AI sentiment",
            "Human majority topic", "Match?", "Human majority sentiment", "Match?"],
       [5, 60, 10, 12, 12, 14, 9, 16, 9])
for k, it in enumerate(items):
    row = FIRST + k
    ai.cell(row=row, column=1, value=k + 1)
    ai.cell(row=row, column=2, value=it[1]).alignment = WRAP
    ai.cell(row=row, column=3, value=it[4])
    ai.cell(row=row, column=4, value=it[5] or "-")
    ai.cell(row=row, column=5, value=SENT_MAP[it[6]])
    ai.cell(row=row, column=6, value=f"='3_Agreement'!{T['majority_col']}{row}")
    ai.cell(row=row, column=7, value=f'=IF(F{row}="","",IF(F{row}=C{row},1,0))')
    ai.cell(row=row, column=8, value=f"='3_Agreement'!{S['majority_col']}{row}")
    ai.cell(row=row, column=9, value=f'=IF(H{row}="","",IF(H{row}=E{row},1,0))')
    ai.row_dimensions[row].height = 36
ai.cell(row=LAST + 2, column=2, value="Human-majority vs AI agreement").font = Font(bold=True)
ai.cell(row=LAST + 2, column=7, value=f'=IF(COUNT(G{FIRST}:G{LAST})=0,"",AVERAGE(G{FIRST}:G{LAST}))').number_format = "0%"
ai.cell(row=LAST + 2, column=9, value=f'=IF(COUNT(I{FIRST}:I{LAST})=0,"",AVERAGE(I{FIRST}:I{LAST}))').number_format = "0%"
ai.cell(row=LAST + 3, column=2,
        value="AI labels: Claude with the codebook prompt (prompts.md, Prompt B). 'none' = no human majority.").font = Font(italic=True)
red = PatternFill("solid", fgColor="F8D7DA")
green = PatternFill("solid", fgColor="D4EDDA")
for col in ("G", "I"):
    ai.conditional_formatting.add(f"{col}{FIRST}:{col}{LAST}", CellIsRule(operator="equal", formula=["0"], fill=red))
    ai.conditional_formatting.add(f"{col}{FIRST}:{col}{LAST}", CellIsRule(operator="equal", formula=["1"], fill=green))
ai.sheet_state = "hidden"  # unhide during the reveal (right-click a tab > Unhide)

out = os.path.join(ROOT, "annotation_worksheet.xlsx")
wb.save(out)
print("saved", out)
