from PIL import Image, ImageDraw, ImageFont


W, H = 1400, 820
OUT = "docs/socme_diagram.png"


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


img = Image.new("RGB", (W, H), "#f8fafc")
d = ImageDraw.Draw(img)

title = font(34, True)
subtitle = font(18)
phase = font(24, True)
label = font(14, True)
small = font(14)
pillfont = font(13, True)


def rounded_box(xy, fill, outline, width=2, radius=14):
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def arrow(x1, y1, x2, y2, color="#64748b", width=3):
    d.line((x1, y1, x2, y2), fill=color, width=width)
    if x2 >= x1:
        pts = [(x2, y2), (x2 - 13, y2 - 7), (x2 - 13, y2 + 7)]
    else:
        pts = [(x2, y2), (x2 + 13, y2 - 7), (x2 + 13, y2 + 7)]
    d.polygon(pts, fill=color)


def down_arrow(x, y1, y2):
    d.line((x, y1, x, y2), fill="#94a3b8", width=2)
    d.polygon([(x, y2), (x - 6, y2 - 10), (x + 6, y2 - 10)], fill="#94a3b8")


def text_lines(x, y, lines, fnt=small, fill="#334155", line_gap=22):
    for i, line in enumerate(lines):
        d.text((x, y + i * line_gap), line, font=fnt, fill=fill)


d.text((70, 60), "SOCME Diagram for VeriSci: AI Research Scientist", font=title, fill="#0f172a")
d.text((70, 96), "Search, Organize, Chat, Map, and Export workflow for Scopus-backed research assistance", font=subtitle, fill="#475569")

pill_data = [
    (70, "S: Search Papers", 175),
    (265, "O: Organize Workspace", 190),
    (475, "C: Chat & Analyze", 178),
    (673, "M: Map Knowledge Graph", 210),
    (903, "E: Export Output", 165),
]
for x, txt, w in pill_data:
    rounded_box((x, 135, x + w, 171), "#e0f2fe", "#7dd3fc", width=1, radius=18)
    d.text((x + 22, 145), txt, font=pillfont, fill="#075985")

boxes = [
    (70, 230, 280, 380, "1. Search", "Query academic sources", ["Scopus, IEEE, OpenAlex,", "Semantic Scholar, arXiv"]),
    (330, 230, 550, 380, "2. Organize", "Save selected papers", ["Workspace, tags, notes,", "paper library"]),
    (600, 230, 820, 380, "3. Chat", "Ask questions on papers", ["RAG over uploaded PDFs,", "summaries, comparison"]),
    (870, 230, 1090, 380, "4. Map", "Build research graph", ["Papers, authors, fields,", "journals, keywords"]),
    (1140, 230, 1350, 380, "5. Export", "Generate report outputs", ["BibTeX, CSV, JSON,", "research gap notes"]),
]
for x1, y1, x2, y2, head, sub, lines in boxes:
    rounded_box((x1, y1, x2, y2), "#ffffff", "#2563eb", width=3)
    d.text((x1 + 25, y1 + 34), head, font=phase, fill="#0f172a")
    d.text((x1 + 25, y1 + 70), sub, font=font(15), fill="#475569")
    text_lines(x1 + 25, y1 + 100, lines)

for x1, x2 in [(280, 330), (550, 600), (820, 870), (1090, 1140)]:
    arrow(x1, 305, x2 - 3, 305)

support = [
    (80, 470, 330, 590, "#eff6ff", "#93c5fd", "Verified Metadata Layer", ["Scopus verification, DOI,", "citations, journal metrics,", "open-access status"]),
    (390, 470, 650, 590, "#ecfdf5", "#86efac", "Deep Research Layer", ["Legal OA PDF resolver,", "temporary download, section", "extraction, deletion"]),
    (710, 470, 970, 590, "#f5f3ff", "#c4b5fd", "AI Analysis Layer", ["Ollama RAG, summaries,", "research gap finder,", "evidence confidence"]),
    (1030, 470, 1290, 590, "#fff7ed", "#fdba74", "Persistence Layer", ["SQLite workspaces, saved", "papers, Neo4j graph,", "RAG document store"]),
]
for x1, y1, x2, y2, fill, outline, head, lines in support:
    rounded_box((x1, y1, x2, y2), fill, outline, width=2, radius=10)
    d.text((x1 + 25, y1 + 32), head, font=label, fill="#0f172a")
    text_lines(x1 + 25, y1 + 62, lines)

for x in [180, 440, 710, 980, 1245]:
    down_arrow(x, 380, 470)

rounded_box((160, 660, 1240, 745), "#ffffff", "#cbd5e1", width=2, radius=12)
d.text((190, 690), "Final User Outputs", font=label, fill="#0f172a")
d.text(
    (190, 720),
    "Curated paper library • Scopus-verified metadata • Evidence-backed research gaps • Knowledge graph insights • Citation exports • Report-ready summaries",
    font=small,
    fill="#334155",
)
for x in [205, 520, 840, 1160]:
    down_arrow(x, 590, 660)

d.text(
    (70, 790),
    "Note: Deep Research downloads only legal open-access PDFs temporarily and deletes files after evidence extraction.",
    font=small,
    fill="#334155",
)

img.save(OUT)
print(OUT)
