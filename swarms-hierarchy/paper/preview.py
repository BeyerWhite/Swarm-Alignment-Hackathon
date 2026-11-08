"""Approximate PDF preview of paper.md (Letter, 1in margins, 10.5pt Arial-like) via pandoc + headless Chrome.
Reports page counts for the main body (through Conclusion + Code and Data) and the full document."""
import re, subprocess
from pathlib import Path
P = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
md = (P / "paper.md").read_text()
title = re.search(r"^TITLE: (.*)$", md, re.M).group(1)
ab = re.search(r"^ABSTRACT: (.*?)\n\n", md, re.S | re.M).group(1)
body = "## " + md.split("\n## ", 1)[1]
head = (f'<h1 class="title">{title}</h1><div class="authors"><b>Sara Ansari</b> (University of Texas at Austin) · <b>Beyer White</b> (Independent)'
        f'<br><br>With Apart Research</div><p><b>Abstract</b></p><div class="abstract">{ab}</div>\n\n')
main = body.split("## Code and Data", 1)[0]
pages = {}
for name, text in (("main", head + main), ("full", head + body)):
    (P / f"preview_{name}.md").write_text(text)
    subprocess.run(["pandoc", f"preview_{name}.md", "-f", "markdown+pipe_tables+implicit_figures", "-s", "--mathml", "--css",
                    "preview.css", "--metadata", "title= ", "-o", f"preview_{name}.html"], cwd=P, check=True)
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000",
                    f"--print-to-pdf={P / f'preview_{name}.pdf'}", f"file://{P / f'preview_{name}.html'}"],
                   capture_output=True, timeout=150)
    pages[name] = len(re.findall(rb"/Type\s*/Page[^s]", (P / f"preview_{name}.pdf").read_bytes()))
print(pages)
