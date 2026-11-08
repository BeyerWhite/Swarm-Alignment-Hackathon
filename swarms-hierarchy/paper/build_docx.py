"""Build the submission .docx from the user's copy of the official Apart template and paper/paper.md.

Every paragraph we write clones the template's own formatting: Heading 2 / Heading 3 paragraphs (Old Standard TT,
14 pt / 13 pt bold, grey), body text (Old Standard TT 11 pt), the template's real Word lists (numbered list for
contributions, numbered list for references, bulleted list elsewhere). The title block, author table, abstract box
and footnote are the template's own; only their text changes. The 'How to use this template' box and all guidance
text are removed. Equations ($...$, $$...$$) become native Word math (OMML) via pandoc.
"""
import copy, re, subprocess, tempfile, zipfile
from pathlib import Path
import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper" / "paper.md"
TEMPLATE = ROOT / "scratch" / "template" / "user_template.docx"
OUT = ROOT / "paper" / "Swarm_Hierarchy_Apart_Report.docx"
AUTHORS = [("Sara Ansari", "University of Texas at Austin"), ("Beyer White", "Independent")]

md = SRC.read_text()
title = re.search(r"^TITLE: (.*)$", md, re.M).group(1).strip()
abstract = re.search(r"^ABSTRACT: (.*?)\n\n", md, re.S | re.M).group(1).strip()
body = "## " + md.split("\n## ", 1)[1]

d = docx.Document(TEMPLATE)
t0 = d.tables[0]


# ---------------------------------------------------------------- prototypes from the template
def proto(prefix):
    for p in d.paragraphs:
        if p.text.strip().startswith(prefix):
            return copy.deepcopy(p._p.find(qn("w:pPr"))), copy.deepcopy(p.runs[0]._r.find(qn("w:rPr")))
    raise KeyError(prefix)


def strip(rpr, *tags):
    if rpr is None:
        return rpr
    for t in tags:
        for e in rpr.findall(qn(f"w:{t}")):
            rpr.remove(e)
    return rpr


H2_PPR, H2_RPR = proto("1. Introduction")
H3_PPR, H3_RPR = proto("Limitations")
BODY_PPR, BODY_RPR = proto("What problem are you")
strip(BODY_RPR, "i", "iCs"); strip(BODY_PPR.find(qn("w:rPr")), "i", "iCs")
NUM_PPR, _ = proto("[First contribution")
REF_PPR, _ = proto("[Reference 1]")
BUL_PPR, _ = proto("Code repository")


# ---------------------------------------------------------------- title, authors, abstract
for p in t0.rows[0].cells[0].paragraphs:
    for r in p.runs:
        if "PROJECT TITLE" in r.text:
            r.text = title

done = set()
for tbl in t0._tbl.iter(qn("w:tbl")):
    tcs = [tc for tr in tbl.findall(qn("w:tr")) for tc in tr.findall(qn("w:tc"))]
    if any(tc.find(qn("w:tbl")) is not None for tc in tcs) or id(tbl) in done:
        continue
    texts = ["".join(t.text or "" for t in tc.iter(qn("w:t"))) for tc in tcs]
    if not any(x.startswith("Author name 1") for x in texts):
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
        rpr = copy.deepcopy(par.runs[0]._r.find(qn("w:rPr"))) if par.runs else None
        for r in par.runs:
            r._r.getparent().remove(r._r)
        for extra in ps[1:]:
            tc.remove(extra)
        r1 = par.add_run(name)
        if rpr is not None:
            r1._r.insert(0, copy.deepcopy(rpr))
        r1.bold = True
        if aff:
            r1.add_break()
            r2 = par.add_run(aff)
            if rpr is not None:
                r2._r.insert(0, copy.deepcopy(rpr))
            r2.bold = False

for cell in t0.rows[1].cells:
    for nt in cell.tables:
        for row in nt.rows:
            for cc in row.cells:
                for p in cc.paragraphs:
                    if p.text.strip().startswith("Summarize your project"):
                        keep = p.runs[0]
                        for r in p.runs[1:]:
                            r._r.getparent().remove(r._r)
                        keep.text = abstract; keep.italic = False; keep.underline = False
                        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

# remove everything after the title block (guidance box and guidance text), keep sectPr
bodyel = d.element.body
after = False
for el in list(bodyel):
    if el is t0._tbl:
        after = True; continue
    if after and el.tag != qn("w:sectPr"):
        bodyel.remove(el)
SECTPR = bodyel.find(qn("w:sectPr"))


# ---------------------------------------------------------------- math (OMML via one pandoc call)
snips = []
for mm in re.finditer(r"\$\$(.+?)\$\$|\$([^$\n]+)\$", body):
    tex = mm.group(1) or mm.group(2)
    if tex not in snips:
        snips.append(tex)
OMML = {}
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
if snips:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "m.md"; dst = Path(td) / "m.docx"
        src.write_text("\n\n".join(f"${s}$" for s in snips) + "\n")
        subprocess.run(["pandoc", str(src), "-o", str(dst)], check=True)
        xml = etree.fromstring(zipfile.ZipFile(dst).read("word/document.xml"))
        paras = [pp for pp in xml.iter(qn("w:p")) if pp.find(f".//{M}oMath") is not None]
        assert len(paras) == len(snips), (len(paras), len(snips))
        for s, pp in zip(snips, paras):
            OMML[s] = pp.findall(f"{M}oMath") or pp.findall(f".//{M}oMath")


# ---------------------------------------------------------------- paragraph factory using template formatting
def new_par(ppr):
    p = OxmlElement("w:p")
    p.append(copy.deepcopy(ppr))
    SECTPR.addprevious(p)
    return Paragraph(p, d._body)


def run(par, text, rpr=None, bold=False, italic=False, size=None):
    r = par.add_run(text)
    r._r.insert(0, copy.deepcopy(rpr if rpr is not None else BODY_RPR))
    if bold:
        r.bold = True
    if italic:
        r.italic = True
    if size:
        r.font.size = Pt(size)
    return r


def inline(par, text, size=None, bold=False, italic=False):
    for tk in re.split(r"(\$[^$\n]+\$|\*\*.+?\*\*|\*[^*\s][^*]*\*|`[^`]+`)", text):
        if not tk:
            continue
        if tk.startswith("$") and tk.endswith("$") and len(tk) > 1:
            for om in OMML[tk[1:-1]]:
                par._p.append(copy.deepcopy(om))
        elif tk.startswith("**") and tk.endswith("**"):
            inline(par, tk[2:-2], size, True, italic)
        elif tk.startswith("*") and tk.endswith("*") and len(tk) > 2:
            inline(par, tk[1:-1], size, bold, True)
        elif tk.startswith("`"):
            r = run(par, tk[1:-1], size=size, bold=bold, italic=italic); r.font.name = "Courier New"
        else:
            run(par, tk, size=size, bold=bold, italic=italic)


def heading(text, level):
    p = new_par(H2_PPR if level == 2 else H3_PPR)
    run(p, text, H2_RPR if level == 2 else H3_RPR)


def list_item(text, kind, level=0):
    ppr = copy.deepcopy({"num": NUM_PPR, "ref": REF_PPR, "bul": BUL_PPR}[kind])
    ppr.find(qn("w:numPr")).find(qn("w:ilvl")).set(qn("w:val"), str(level))
    ind = ppr.find(qn("w:ind"))
    if ind is not None:
        ind.set(qn("w:left"), str(720 + 360 * level)); ind.set(qn("w:hanging"), "360")
        for a in ("right",):
            if ind.get(qn(f"w:{a}")) is not None:
                del ind.attrib[qn(f"w:{a}")]
    p = new_par(ppr)
    p.paragraph_format.space_after = Pt(2)
    inline(p, text)


def borders(cell):
    tcPr = cell._tc.get_or_add_tcPr(); b = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "999999")
        b.append(e)
    tcPr.append(b)


# ---------------------------------------------------------------- write the body
lines = body.split("\n")
section = ""
i = 0
while i < len(lines):
    ln = lines[i].rstrip()
    if not ln.strip():
        i += 1; continue
    if ln.startswith("### "):
        heading(ln[4:].strip(), 3); i += 1; continue
    if ln.startswith("## "):
        section = ln[3:].strip(); heading(section, 2); i += 1; continue
    md_ = re.match(r"^\$\$(.+?)\$\$\s*$", ln)
    if md_:
        p = new_par(BODY_PPR); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(3); p.paragraph_format.space_after = Pt(6)
        for om in OMML[md_.group(1)]:
            p._p.append(copy.deepcopy(om))
        i += 1; continue
    mf = re.match(r"!\[(.*)\]\((.*?)\)(\{width=([\d.]+)in\})?", ln)
    if mf:
        cap, path = mf.group(1), mf.group(2)
        width = float(mf.group(4)) if mf.group(4) else 6.0
        p = new_par(BODY_PPR); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(ROOT / "paper" / path), width=Inches(width))
        mc = re.match(r"(Figure [A-Z]?\d+\.)\s*(.*)", cap)
        cp = new_par(BODY_PPR); cp.paragraph_format.space_after = Pt(8)
        run(cp, mc.group(1) + " ", bold=True, size=9.5)
        inline(cp, mc.group(2), size=9.5)
        i += 1; continue
    if ln.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            if not re.match(r"^\|\s*-", lines[i]):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
            i += 1
        tb = d.add_table(rows=len(rows), cols=len(rows[0]))
        SECTPR.addprevious(tb._tbl)
        tb.alignment = 1
        for ri, row in enumerate(rows):
            for ci, val in enumerate(row):
                cell = tb.cell(ri, ci); borders(cell)
                cp = cell.paragraphs[0]
                cp._p.insert(0, copy.deepcopy(BODY_PPR))
                inline(cp, val, size=9.5, bold=(ri == 0))
        sp = new_par(BODY_PPR); sp.paragraph_format.space_after = Pt(2)
        continue
    ml = re.match(r"^(\s*)(\d+\.|-) (.*)$", ln)
    if ml:
        level = len(ml.group(1)) // 2
        numbered = ml.group(2) != "-"
        kind = "ref" if section.startswith("References") else ("num" if numbered else "bul")
        list_item(ml.group(3), kind, level); i += 1; continue
    if ln.startswith("*Table"):
        p = new_par(BODY_PPR); inline(p, ln.strip("*"), size=9.5, italic=True); i += 1; continue
    p = new_par(BODY_PPR); p.paragraph_format.space_after = Pt(4)
    inline(p, ln); i += 1

d.save(OUT)
print("wrote", OUT)
