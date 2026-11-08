"""Build the submission .docx from the official Apart template (scratch/template/template.docx) and paper/paper.md.
Keeps the template's title block, styles and footnote; replaces guidance text; removes the 'How to use' box."""
import re, copy, sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper" / "paper.md"
OUT = ROOT / "paper" / "Swarm_Hierarchy_Apart_Report.docx"
AUTHORS = [("Sara Ansari", "University of Texas at Austin"), ("Beyer White", "Independent")]   # authors 2-6 left blank for the team to fill

md = SRC.read_text()
title = re.search(r"^TITLE: (.*)$", md, re.M).group(1).strip()
abstract = re.search(r"^ABSTRACT: (.*?)\n\n", md, re.S | re.M).group(1).strip()
body = md.split("\n## ", 1)[1]
body = "## " + body

d = docx.Document(ROOT / "scratch" / "template" / "template.docx")
t0 = d.tables[0]


def set_runs(p, text, italic=None, bold=None):
    runs = p.runs
    keep = runs[0]
    keep.text = text
    if italic is not None:
        keep.italic = italic
    if bold is not None:
        keep.bold = bold
    for r in runs[1:]:
        r._element.getparent().remove(r._element)


# --- title (keep footnote reference run)
for p in t0.rows[0].cells[0].paragraphs:
    for r in p.runs:
        if "PROJECT TITLE" in r.text:
            r.text = title

# --- authors: find the 2x3 author table and rewrite each cell's paragraph wholesale (runs are fragmented)
from docx.text.paragraph import Paragraph
done = set()
for tbl in t0._tbl.iter(qn("w:tbl")):
    tcs = [tc for tr in tbl.findall(qn("w:tr")) for tc in tr.findall(qn("w:tc"))]   # direct cells only
    if any(tc.find(qn("w:tbl")) is not None for tc in tcs):                        # skip tables that nest others
        continue
    texts = ["".join(t.text or "" for t in tc.iter(qn("w:t"))) for tc in tcs]
    if not any(x.startswith("Author name 1") for x in texts) or id(tbl) in done:
        continue
    done.add(id(tbl))
    k = 0
    for tc, x in zip(tcs, texts):
        if not x.startswith("Author name"):
            continue
        name, aff = AUTHORS[k] if k < len(AUTHORS) else ("", "")
        k += 1
        ps = tc.findall(qn("w:p"))
        par = Paragraph(ps[0], None)
        runs = par.runs
        rpr = copy.deepcopy(runs[0]._element.find(qn("w:rPr"))) if runs and runs[0]._element.find(qn("w:rPr")) is not None else None
        for r in runs:
            r._element.getparent().remove(r._element)
        for extra in ps[1:]:
            tc.remove(extra)
        r1 = par.add_run(name); r1.bold = True
        if aff:
            r1.add_break(); par.add_run(aff)

for cell in t0.rows[1].cells:
    for nt in cell.tables:
        for row in nt.rows:
            for cc in row.cells:
                for p in cc.paragraphs:
                    if p.text.strip().startswith("Summarize your project"):
                        set_runs(p, abstract, italic=False)
                        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

# --- remove everything after the title table (info box + guidance), keep sectPr
bodyel = d.element.body
started = False
for el in list(bodyel):
    if el is t0._tbl:
        started = True
        continue
    if started and el.tag != qn("w:sectPr"):
        bodyel.remove(el)


# --- math: convert every $...$ / $$...$$ snippet to native Word OMML (one pandoc call, cached)
import subprocess, tempfile, zipfile
from lxml import etree
snips = []
for mm in re.finditer(r"\$\$(.+?)\$\$|\$([^$\n]+)\$", body):
    tex = mm.group(1) or mm.group(2)
    if tex not in snips:
        snips.append(tex)
OMML = {}
if snips:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "m.md"; dst = Path(td) / "m.docx"
        src.write_text("\n\n".join(f"${s}$" for s in snips) + "\n")
        subprocess.run(["pandoc", str(src), "-o", str(dst)], check=True)
        xml = etree.fromstring(zipfile.ZipFile(dst).read("word/document.xml"))
        M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
        paras = [pp for pp in xml.iter(qn("w:p")) if pp.find(f".//{M}oMath") is not None]
        assert len(paras) == len(snips), (len(paras), len(snips))
        for s, pp in zip(snips, paras):
            OMML[s] = pp.findall(f"{M}oMath") or pp.findall(f".//{M}oMath")


def add_math(p, tex):
    for om in OMML[tex]:
        p._p.append(copy.deepcopy(om))


# --- markdown-subset writer
def add_inline(p, text, size=None, italic_all=False, bold=False, italic=False):
    toks = re.split(r"(\$[^$\n]+\$|\*\*.+?\*\*|\*[^*\s][^*]*\*|`[^`]+`)", text)
    for tk in toks:
        if not tk:
            continue
        if tk.startswith("$") and tk.endswith("$") and len(tk) > 1:
            add_math(p, tk[1:-1]); continue
        if tk.startswith("**") and tk.endswith("**"):
            add_inline(p, tk[2:-2], size, italic_all, True, italic); continue
        if tk.startswith("*") and tk.endswith("*") and len(tk) > 2:
            add_inline(p, tk[1:-1], size, italic_all, bold, True); continue
        if tk.startswith("`"):
            r = p.add_run(tk[1:-1]); r.font.name = "Courier New"
        else:
            r = p.add_run(tk)
        r.bold = bold or None
        r.italic = (italic or italic_all) or None
        if size:
            r.font.size = Pt(size)


def para(text, style="normal", size=None, align=None, italic_all=False, space_after=4):
    p = d.add_paragraph(style=style)
    add_inline(p, text, size, italic_all)
    p.paragraph_format.space_after = Pt(space_after)
    if align:
        p.alignment = align
    return p


def set_cell_border(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    b = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "999999")
        b.append(e)
    tcPr.append(b)


lines = body.split("\n")
i = 0
fig_n = 0
while i < len(lines):
    ln = lines[i].rstrip()
    if not ln.strip():
        i += 1; continue
    if ln.startswith("### "):
        d.add_paragraph(ln[4:].strip(), style="Heading 3"); i += 1; continue
    if ln.startswith("## "):
        h = ln[3:].strip()
        d.add_paragraph(h, style="Heading 2"); i += 1; continue
    mdisp = re.match(r"^\$\$(.+?)\$\$\s*$", ln)
    if mdisp:
        dp = d.add_paragraph(); dp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        dp.paragraph_format.space_before = Pt(3); dp.paragraph_format.space_after = Pt(6)
        add_math(dp, mdisp.group(1))
        i += 1; continue
    m = re.match(r"!\[(.*)\]\((.*?)\)(\{width=([\d.]+)in\})?", ln)
    if m:
        cap, path = m.group(1), m.group(2)
        width = float(m.group(4)) if m.group(4) else 6.3
        pic = d.add_paragraph(); pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pic.add_run().add_picture(str(ROOT / "paper" / path), width=Inches(width))
        mm = re.match(r"(Figure [A-Z]?\d+\.)\s*(.*)", cap)
        cp = d.add_paragraph(); cp.paragraph_format.space_after = Pt(8)
        r = cp.add_run(mm.group(1) + " "); r.bold = True; r.font.size = Pt(9)
        add_inline(cp, mm.group(2), size=9)
        i += 1; continue
    if ln.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            if not re.match(r"^\|\s*-", lines[i]):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
            i += 1
        tb = d.add_table(rows=len(rows), cols=len(rows[0]))
        tb.alignment = 1
        for ri, row in enumerate(rows):
            for ci, val in enumerate(row):
                cell = tb.cell(ri, ci); set_cell_border(cell)
                cp = cell.paragraphs[0]; add_inline(cp, val, size=9)
                if ri == 0:
                    for r in cp.runs:
                        r.bold = True
        d.add_paragraph().paragraph_format.space_after = Pt(2)
        continue
    if re.match(r"^\s*(\d+\. |- )", ln):
        indent = (len(ln) - len(ln.lstrip())) // 2
        s = ln.lstrip()
        num = re.match(r"^(\d+)\. ", s)
        txt = s[len(num.group(0)):] if num else s[2:]
        p = d.add_paragraph(style="normal")
        p.paragraph_format.left_indent = Inches(0.3 + 0.25 * indent); p.paragraph_format.first_line_indent = Inches(-0.2)
        p.paragraph_format.space_after = Pt(2)
        p.add_run((num.group(1) + ". ") if num else ("◦ " if indent else "• "))
        add_inline(p, txt)
        i += 1; continue
    if ln.startswith("*Table"):
        para(ln.strip("*"), size=9, italic_all=True); i += 1; continue
    para(ln); i += 1

d.save(OUT)
print("wrote", OUT)
