"""Build the GalaxyCLIP-3D project report (PDF) with ReportLab, from numbers.json and the report figures.

    cd GalaxyCLIP_3D && PYTHONPATH=~/tools/report_pylibs:pylibs:. python -s report/build_pdf.py
    # -> outputs/project_report/GalaxyCLIP3D_report.pdf

Every number in the text comes from outputs/project_report/numbers.json (report/collect_numbers.py); figures from
report/figures.py. Equations are rendered with matplotlib's mathtext (no LaTeX installation needed).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                      # noqa: E402
from PIL import Image as PILImage                                                    # noqa: E402
from reportlab.lib import colors                                                     # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT                       # noqa: E402
from reportlab.lib.fonts import addMapping                                           # noqa: E402
from reportlab.lib.pagesizes import A4                                               # noqa: E402
from reportlab.lib.styles import ParagraphStyle                                      # noqa: E402
from reportlab.lib.units import cm                                                   # noqa: E402
from reportlab.pdfbase import pdfmetrics                                             # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont                                         # noqa: E402
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether,  # noqa: E402
                                LongTable, NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents                      # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/project_report"
FIG = OUT / "figures"
EQ = OUT / "equations"
PDF = OUT / "GalaxyCLIP3D_report.pdf"
TITLE = Path(__file__).with_name("title.local.json")   # author and affiliation; not in git (report/README.md)
_who = json.loads(TITLE.read_text()) if TITLE.exists() else {}
AUTHOR, PLACE, DATE = _who.get("author", ""), _who.get("affiliation", ""), "18 November 2026"
REPO_URL = "https://github.com/Apoptoxin-4869-APTX/GalaxyCLIP-3D"

INK, INK2, MUTED, RULE, TINT, ACCENT = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#f3f2ee", "#1f3a5f"
PAGE_W, PAGE_H = A4
LM = RM = 2.2 * cm
TM, BM = 2.4 * cm, 2.2 * cm
TEXT_W = PAGE_W - LM - RM


# ---------------------------------------------------------------- fonts and styles
def register_fonts() -> None:
    mpl = Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf"
    dv = Path("/usr/share/fonts/truetype/dejavu")
    fams = {"STIX": (mpl / "STIXGeneral.ttf", mpl / "STIXGeneralBol.ttf", mpl / "STIXGeneralItalic.ttf",
                     mpl / "STIXGeneralBolIta.ttf"),
            "DVS": (dv / "DejaVuSans.ttf", dv / "DejaVuSans-Bold.ttf", dv / "DejaVuSans-Oblique.ttf",
                    dv / "DejaVuSans-BoldOblique.ttf"),
            "DVSC": (dv / "DejaVuSansCondensed.ttf", dv / "DejaVuSansCondensed-Bold.ttf",
                     dv / "DejaVuSansCondensed-Oblique.ttf", dv / "DejaVuSansCondensed-BoldOblique.ttf")}
    for fam, (r, b, i, bi) in fams.items():
        for suffix, path in (("", r), ("-Bold", b), ("-Italic", i), ("-BoldItalic", bi)):
            pdfmetrics.registerFont(TTFont(fam + suffix, str(path)))
        addMapping(fam, 0, 0, fam)
        addMapping(fam, 1, 0, fam + "-Bold")
        addMapping(fam, 0, 1, fam + "-Italic")
        addMapping(fam, 1, 1, fam + "-BoldItalic")


register_fonts()
S = {
    "body": ParagraphStyle("body", fontName="STIX", fontSize=10.6, leading=14.6, alignment=TA_JUSTIFY,
                           spaceAfter=6.5, textColor=INK),
    "lead": ParagraphStyle("lead", fontName="STIX", fontSize=11.4, leading=15.6, alignment=TA_LEFT,
                           spaceAfter=8, textColor=INK),
    "h1": ParagraphStyle("h1", fontName="DVS-Bold", fontSize=15, leading=19, spaceBefore=6, spaceAfter=8,
                         textColor=ACCENT),
    "h2": ParagraphStyle("h2", fontName="DVS-Bold", fontSize=11.4, leading=15, spaceBefore=9, spaceAfter=4,
                         textColor=INK),
    "h3": ParagraphStyle("h3", fontName="DVS-Bold", fontSize=9.6, leading=13, spaceBefore=5, spaceAfter=2,
                         textColor=INK2),
    "caption": ParagraphStyle("caption", fontName="DVS", fontSize=8.1, leading=10.8, textColor=INK2,
                              spaceBefore=3, spaceAfter=11, alignment=TA_LEFT),
    "note": ParagraphStyle("note", fontName="DVS", fontSize=7.4, leading=9.6, textColor=MUTED, spaceBefore=2,
                           spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="DVSC", fontSize=7.8, leading=9.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DVSC-Bold", fontSize=7.8, leading=9.6, textColor=INK),
    "bullet": ParagraphStyle("bullet", fontName="STIX", fontSize=10.6, leading=14.4, leftIndent=14,
                             bulletIndent=3, spaceAfter=3.2, alignment=TA_LEFT, textColor=INK),
    "box": ParagraphStyle("box", fontName="STIX", fontSize=10, leading=13.6, textColor=INK, spaceAfter=4,
                          alignment=TA_LEFT),
    "boxh": ParagraphStyle("boxh", fontName="DVS-Bold", fontSize=9.4, leading=12.5, textColor=ACCENT,
                           spaceAfter=4),
    "toc0": ParagraphStyle("toc0", fontName="DVS-Bold", fontSize=10, leading=15, leftIndent=0),
    "toc1": ParagraphStyle("toc1", fontName="DVS", fontSize=9, leading=12.5, leftIndent=16, textColor=INK2),
}


# ---------------------------------------------------------------- document with a table of contents
class Doc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(str(path), pagesize=A4, leftMargin=LM, rightMargin=RM, topMargin=TM, bottomMargin=BM,
                         title="GalaxyCLIP-3D: project report", author=AUTHOR,
                         subject="Contrastive alignment of TNG50 galaxies as 3D point clouds", **kw)
        frame = Frame(LM, BM, TEXT_W, PAGE_H - TM - BM, id="f", leftPadding=0, rightPadding=0, topPadding=0,
                      bottomPadding=0)
        self.addPageTemplates([PageTemplate("title", [frame], onPage=lambda c, d: None),
                               PageTemplate("body", [frame], onPage=self.decorate)])

    @staticmethod
    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont("DVS", 7.6)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(LM, PAGE_H - 1.45 * cm, "GalaxyCLIP-3D · project report")
        canvas.drawRightString(PAGE_W - RM, PAGE_H - 1.45 * cm, DATE)
        canvas.setStrokeColor(colors.HexColor(RULE))
        canvas.setLineWidth(0.4)
        canvas.line(LM, PAGE_H - 1.6 * cm, PAGE_W - RM, PAGE_H - 1.6 * cm)
        canvas.drawCentredString(PAGE_W / 2, 1.2 * cm, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, f):
        level = getattr(f, "toc_level", None)
        if level is not None:
            self.notify("TOCEntry", (level, f.getPlainText(), self.page))


# ---------------------------------------------------------------- building blocks
class Builder:
    def __init__(self, n: dict):
        self.n, self.story, self.fig_no, self.tab_no = n, [], 0, 0

    def add(self, *flow):
        self.story.extend(flow)

    def p(self, text, style="body"):
        self.add(Paragraph(text, S[style]))

    def h1(self, text):
        self.add(CondPageBreak(6 * cm))
        para = Paragraph(text, S["h1"])
        para.toc_level = 0
        self.add(para)

    def h2(self, text):
        self.add(CondPageBreak(3.5 * cm))
        para = Paragraph(text, S["h2"])
        para.toc_level = 1
        self.add(para)

    def h3(self, text):
        self.add(Paragraph(text, S["h3"]))

    def bullets(self, items, style="bullet"):
        for it in items:
            self.add(Paragraph(it, S[style], bulletText="•"))
        self.add(Spacer(1, 3))

    def figure(self, name, caption, width=TEXT_W):
        path = FIG / name
        w, h = PILImage.open(path).size
        self.fig_no += 1
        img = Image(str(path), width=width, height=width * h / w)
        img.hAlign = "CENTER"
        self.add(KeepTogether([Spacer(1, 4), img, Paragraph(f"<b>Figure {self.fig_no}.</b> {caption}",
                                                              S["caption"])]))
        return self.fig_no

    def equation(self, tex, name, size=15):
        EQ.mkdir(parents=True, exist_ok=True)
        path = EQ / f"{name}.png"
        plt.rcParams["mathtext.fontset"] = "stix"
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${tex}$", fontsize=size)
        fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.03, facecolor="white")
        plt.close(fig)
        w, h = PILImage.open(path).size
        img = Image(str(path), width=w / 400 * 72, height=h / 400 * 72)
        img.hAlign = "CENTER"
        self.add(Spacer(1, 2), img, Spacer(1, 6))

    def box(self, title, paras):
        """Tinted box, one row per paragraph so that it can break across pages."""
        rows = [[Paragraph(title, S["boxh"])]] + [[Paragraph(t, S["box"])] for t in paras]
        t = Table(rows, colWidths=[TEXT_W])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(TINT)),
                               ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(RULE)),
                               ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                               ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                               ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, -1), (-1, -1), 7)]))
        self.add(Spacer(1, 3), t, Spacer(1, 9))

    def table(self, caption, header, rows, widths=None, note="", size=7.8, group_rows=(), bold_rows=(),
              align_right_from=1, long=False):
        """A booktabs-like table: header rule, light group rows, numbers right-aligned."""
        self.tab_no += 1
        cs = ParagraphStyle("c", parent=S["cell"], fontSize=size, leading=size * 1.24)
        cb = ParagraphStyle("cb", parent=S["cellb"], fontSize=size, leading=size * 1.24)
        cr = ParagraphStyle("cr", parent=cs, alignment=2)
        cbr = ParagraphStyle("cbr", parent=cb, alignment=2)
        data = [[Paragraph(h, cbr if j >= align_right_from else cb) for j, h in enumerate(header)]]
        for i, r in enumerate(rows):
            line = []
            for j, c in enumerate(r):
                indent = 0
                while c.startswith("&nbsp;"):
                    c, indent = c[6:], indent + 1
                st = (cbr if j >= align_right_from else cb) if i in bold_rows else \
                    (cr if j >= align_right_from else cs)
                if indent:
                    st = ParagraphStyle("ci", parent=st, leftIndent=3 * indent)
                if c.startswith("*") and c.endswith("*") and len(c) > 2:
                    c = f"<i>{c.strip('*')}</i>"
                line.append(Paragraph(c, st))
            data.append(line)
        if widths is None:
            first = 0.34 if len(header) > 4 else 0.45
            rest = (1 - first) / (len(header) - 1)
            widths = [first] + [rest] * (len(header) - 1)
        widths = [w * TEXT_W for w in widths]
        cls = LongTable if long else Table
        t = cls(data, colWidths=widths, repeatRows=1)
        style = [("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.HexColor(INK)),
                 ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor(INK)),
                 ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.HexColor(INK)),
                 ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.2),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2), ("LEFTPADDING", (0, 0), (-1, -1), 3),
                 ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
        for g in group_rows:
            style += [("BACKGROUND", (0, g + 1), (-1, g + 1), colors.HexColor(TINT)),
                      ("SPAN", (0, g + 1), (-1, g + 1))]
        t.setStyle(TableStyle(style))
        cap = Paragraph(f"<b>Table {self.tab_no}.</b> {caption}", S["caption"])
        parts = [cap, t] + ([Paragraph(note, S["note"])] if note else [Spacer(1, 10)])
        if long:
            self.add(*parts)
        else:
            self.add(KeepTogether(parts))
        return self.tab_no

    def md_table(self, name, caption=None, widths=None, size=7.8, long=False, note=True):
        t = self.n["tables"][name]
        groups = [i for i, r in enumerate(t["rows"]) if r[0].startswith("*") and all(c == "" for c in r[1:])]
        return self.table(caption or t["caption"], t["header"], t["rows"], widths=widths,
                          note=t["note"] if note else "", size=size, group_rows=groups, long=long)


def f1(x):
    return f"{x:.1f}"


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


# ---------------------------------------------------------------- content
def title_page(b: Builder):
    n = b.n
    t = n["test"]
    b.add(Spacer(1, 3.2 * cm))
    b.p('<font name="DVS" size="10" color="#898781">PROJECT REPORT</font>', "lead")
    b.add(Spacer(1, 0.3 * cm))
    b.add(Paragraph("GalaxyCLIP-3D", ParagraphStyle("t", fontName="DVS-Bold", fontSize=30, leading=36,
                                                    textColor=ACCENT)))
    b.add(Spacer(1, 0.25 * cm))
    b.add(Paragraph("Contrastive alignment of stars, dark matter, gas and HI in TNG50 galaxies represented as 3D "
                    "point clouds", ParagraphStyle("st", fontName="DVS", fontSize=15, leading=20, textColor=INK)))
    b.add(Spacer(1, 1.2 * cm))
    facts = [["Data", f"{n['dataset']['selected']:,} galaxies of TNG50-1 at z = 0, four point clouds each"],
             ["Model", "four PointNet encoders, one shared 512-dimensional embedding space"],
             ["Result", f"test R@1 {f1(t['r1_final'])}% (95% interval {f1(t['interval'][0])}–"
                        f"{f1(t['interval'][1])}), chance 0.25%"],
             ["Period", "22 October – 18 November 2026"]]
    tt = Table([[Paragraph(k, S["cellb"]), Paragraph(v, ParagraphStyle("v", parent=S["box"], fontSize=10.4))]
                for k, v in facts], colWidths=[2.4 * cm, TEXT_W - 2.4 * cm])
    tt.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.HexColor(RULE)),
                            ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor(RULE)),
                            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    b.add(tt)
    b.add(Spacer(1, 1.0 * cm))
    b.figure("example_galaxy.png", "One galaxy of the dataset as the model sees it: 2,048 points per modality, "
             "here rotated so that the stars' spin points up (during training every cloud gets its own random "
             "rotation). Colours: stellar age, dark-matter speed, gas temperature and HI rotation velocity.")
    b.add(Spacer(1, 1.2 * cm))
    b.p((f'<font name="DVS" size="11">{AUTHOR}</font><br/>' if AUTHOR else "")
        + f'<font name="DVS" size="9.5" color="#52514e">{PLACE + "<br/>" if PLACE else ""}{DATE}</font>', "lead")
    b.add(NextPageTemplate("body"), PageBreak())


def contents(b: Builder):
    b.add(Paragraph("Contents", S["h1"]))
    toc = TableOfContents()
    toc.levelStyles = [S["toc0"], S["toc1"]]
    toc.dotsMinLevel = 0
    b.add(toc, PageBreak())


def summary(b: Builder):
    n, t = b.n, b.n["test"]
    d, dm = n["dataset"], n["dm_control"]
    rs = d["splits"]
    pm = {r[0]: r for r in n["tables"]["probes_summary"]["rows"]}
    r_c21, r_c29 = dm["retrieval"]["C21-stellar"], dm["retrieval"]["c29-Rvmax"]
    p_c21, p_c29 = dm["probes"]["C21-stellar"]["dm"], dm["probes"]["c29-Rvmax"]["dm"]
    p_s29 = dm["probes"]["stats-Rvmax"]["dm"]
    b.h1("Summary")
    b.p("This project asks whether a neural network can learn one shared representation for four physically "
        "different views of the same simulated galaxy: its stars, its dark matter, its gas and its neutral atomic "
        "hydrogen (HI), each given as a three-dimensional cloud of points drawn from the simulation's particles. "
        "The approach follows CLIP: one encoder per view, trained so that the four clouds of a galaxy land close "
        "together in a common embedding space while clouds of different galaxies are pushed apart.", "lead")
    b.h3("What was done")
    b.bullets([
        f"<b>Dataset.</b> A new, documented point-cloud dataset was built from the raw TNG50-1 simulation at z = 0: "
        f"{d['selected']:,} galaxies, each with stars, dark matter, gas and HI. {d['usable_r5']:,} are usable at the "
        f"default sphere of 5 stellar half-mass radii, split by host halo into {rs['train']['galaxies']:,} / "
        f"{rs['val']['galaxies']} / {rs['test']['galaxies']} galaxies for training, validation and test.",
        "<b>Model and evaluation.</b> Four PointNet encoders are trained with a contrastive (InfoNCE) loss over all "
        "six modality pairs. Retrieval is scored over the full validation or test gallery, and linear probes "
        "measure which galaxy properties the embeddings contain.",
        "<b>Experiments.</b> Baselines without learning, a control for an orientation shortcut, a first comparison "
        f"of PointNet and a patch transformer, {n['facts']['phase_c_runs']} ablation runs, a final setup trained "
        f"with three seeds and scored once on the test set, and two controls on what the dark-matter embedding "
        f"encodes.",
    ])
    b.h3("Main results")
    b.bullets([
        f"<b>Retrieval.</b> The final setup finds the matching cloud of another modality as the single best match "
        f"(R@1) for <b>{f1(t['r1_final'])}%</b> of test queries (mean over the 10 search directions without the "
        f"near-duplicate gas–HI pair; 95% interval {f1(t['interval'][0])}–{f1(t['interval'][1])}). Chance is "
        f"0.25%, hand-made statistics with CCA reach 3.9%, and the first PointNet {f1(t['r1_base'])}% "
        f"(improvement {f1(t['final_minus_base'])} points, interval {f1(t['diff_interval'][0])}–"
        f"{f1(t['diff_interval'][1])}).",
        "<b>An orientation shortcut, found and closed.</b> If a galaxy's four clouds share one orientation, a model "
        "learns to match them by which way the galaxy points: 75.4% test R@1, but only 6.0% once the clouds are "
        "rotated independently. All reported models therefore rotate every cloud on its own.",
        f"<b>What the embeddings know.</b> Linear probes on frozen embeddings predict 18 galaxy properties far "
        f"better than hand-made statistics of the same clouds. The gain is largest for the dark matter: mean test "
        f"R² {pm['final setup (3 seeds)'][2].split(' ')[0]} against {pm['hand-made statistics'][2].split(' ')[0]}; "
        f"the dark-matter cloud alone predicts the stellar mass with R² 0.80 (statistics: 0.29).",
        f"<b>Part of the dark-matter result came from the stars.</b> Every cloud, dark matter included, is cut and "
        f"scaled by the stars' size. Cutting the dark matter at the halo's own radius R<sub>vmax</sub> instead "
        f"halves stars–dark-matter retrieval ({f1(r_c21['pairs']['stars-dm'])}% → "
        f"{f1(r_c29['pairs']['stars-dm'])}% R@1 on the same {dm['shared']['val']} validation galaxies) and lowers "
        f"the stellar-mass R² from {f2(p_c21['props']['log_mstar_total'])} to "
        f"{f2(p_c29['props']['log_mstar_total'])}. The halo-frame embedding still beats statistics of the same "
        f"clouds (mean R² {f2(p_c29['mean_r2'])} against {f2(p_s29['mean_r2'])}).",
    ])
    c30 = n.get("c30")
    if c30:
        rc = c30["retrieval"]["c30-subset"]
        pc = c30["probes"]["c30-subset"]["dm"]
        b.bullets([f"<b>Matched control.</b> The final setup retrained on exactly the galaxies of the halo-frame run "
                   f"reaches stars–dark-matter R@1 {f1(rc['pairs']['stars-dm'])}% and a dark-matter stellar-mass "
                   f"R² of {f2(pc['props']['log_mstar_total'])} on the same galaxies (Section 6.8). "
                   + n.get("c30_summary", "")])
    else:
        b.bullets(["<b>Matched control.</b> A run of the final setup on exactly the galaxies of the halo-frame run "
                   "was still in the queue when this version of the report was built (Section 6.8)."])
    b.h3("Open points")
    b.p("The test set has been used once, as planned, so later changes must be judged on validation. The controls "
        "ran with one seed each. All data come from one snapshot (z = 0). The next research step is the alignment "
        "of these 3D embeddings with the frozen 2D map model of GalaxyCLIP-2D (Section 8).")


def introduction(b: Builder):
    n = b.n
    b.h1("1  Introduction")
    b.p("Large simulations such as IllustrisTNG produce galaxies whose stars, dark matter and gas can each be "
        "inspected in full three-dimensional detail. Observations, in contrast, see different components through "
        "different instruments: starlight in optical images, neutral hydrogen in 21 cm radio maps, dark matter only "
        "indirectly. A representation in which the components of one galaxy are tied together would let a model "
        "move between them: find the dark-matter structure that fits an observed stellar distribution, or ask what "
        "the gas of a galaxy implies about its stars.")
    b.p("GalaxyCLIP-3D studies this with point clouds. Each galaxy is described by four clouds of 2,048 points, one "
        "per component, drawn from the simulation's own particles and carrying positions, velocities and physical "
        "quantities such as stellar age, gas temperature or neutral fraction. Unlike projected maps, point clouds "
        "keep the full 3D structure and the kinematics. The model is CLIP-like: an encoder per modality and a "
        "contrastive loss that brings the four clouds of a galaxy together in one 512-dimensional space.")
    b.p("The project complements GalaxyCLIP-2D, which aligns seven projected maps of the same galaxies. The galaxy "
        "selection matches the 2D maps, and every point cloud records the halo it belongs to, so the two models "
        "can later be aligned with each other. This report covers the 3D work only.")
    b.h3("Contributions")
    b.bullets([
        f"A reproducible point-cloud dataset (v3) of {n['dataset']['selected']:,} TNG50-1 galaxies with four "
        "modalities, splits by host halo that stay consistent when more snapshots are added, quality checks and "
        "full documentation.",
        "A training and evaluation framework for four-modality contrastive alignment with missing modalities "
        "(galaxies without enough gas), full-gallery retrieval, linear probes and bootstrap intervals.",
        f"A systematic study: baselines without learning, an orientation-shortcut control, "
        f"{n['facts']['phase_c_runs']} ablation runs and a final setup with three seeds, scored once on held-out "
        f"test galaxies.",
        "Two controls on what the dark-matter embedding encodes, separating information in the dark matter itself "
        "from information leaked through the way the clouds are scaled.",
    ])
    b.p("Section 2 introduces the astrophysical and machine-learning terms used throughout. Section 3 places the "
        "work among related studies. Sections 4 and 5 describe the data and the method, Section 6 the results, and "
        "Sections 7 and 8 discuss limitations and next steps. The appendices list hyperparameters, full tables, "
        "the code and the project timeline.")


def background(b: Builder):
    b.h1("2  Background for both readers")
    b.p("The project sits between astrophysics and machine learning. The two boxes below give the minimum needed "
        "from each side; readers familiar with one side can skip its box.")
    b.box("Astrophysics in brief", [
        "<b>TNG50-1</b> is the highest-resolution run of the IllustrisTNG cosmological simulations: a cube of about "
        "51.7 Mpc (35 Mpc/h) on a side, evolved from the early universe to today with gravity, gas dynamics, "
        "magnetic fields and models for star formation and black holes. Matter is represented by dark-matter "
        "particles (each 4.5×10<super>5</super> solar masses), gas cells and star particles. A <b>snapshot</b> "
        "stores the state at one time; snapshot 99 is today (redshift z = 0).",
        "A <b>halo</b> is a gravitationally bound clump found by the friends-of-friends algorithm; a <b>subhalo</b>, "
        "found by Subfind inside a halo, is what we call a <b>galaxy</b>. The largest subhalo of a halo is its "
        "<b>central</b>, the others are <b>satellites</b>.",
        "The <b>stellar half-mass radius</b> R½ contains half of a galaxy's stellar mass; it measures the size of "
        "the stellar body. <b>R<sub>vmax</sub></b> is the radius at which the circular velocity, set by the total "
        "mass inside a radius, peaks; it is a property of the halo's mass profile.",
        "<b>Dark matter</b> is collisionless and makes up most of a galaxy's mass. <b>Gas</b> carries density, "
        "temperature, metallicity and magnetic field. <b>HI</b> is neutral atomic hydrogen, the gas seen by 21 cm "
        "radio telescopes; here it is computed per gas cell as 0.76 × neutral fraction × mass.",
    ])
    b.box("Machine learning in brief", [
        "<b>Contrastive learning (CLIP).</b> Encoders map each view to a unit-length vector (an embedding). For a "
        "batch of galaxies, the loss rewards a high similarity between the two views of the same galaxy and a low "
        "similarity to the other galaxies of the batch (the InfoNCE loss). A <b>temperature</b> τ sets how sharply "
        "the softmax over similarities separates them.",
        "<b>Retrieval, R@k.</b> Each validation or test galaxy is used as a query in one modality and searched for "
        "among all galaxies in another modality (the full gallery, not just a batch). R@1 is the fraction of queries "
        "whose own galaxy is the single best match; R@10 counts a hit within the ten best. Chance is one over the "
        "gallery size, about 0.2% here.",
        "<b>Linear probes.</b> A linear (ridge) regression is fitted on frozen embeddings to predict a physical "
        "property, for example stellar mass. R² = 1 is a perfect prediction, R² = 0 is no better than predicting "
        "the mean. For yes/no properties the area under the ROC curve (AUC) is used: 0.5 is guessing.",
        "<b>PointNet</b> applies the same small network to every point and then takes the maximum over points, so "
        "the result does not depend on the order of the points. A <b>patch transformer</b> (as in Point-MAE) groups "
        "points into local patches and runs a transformer over the patch tokens.",
    ])


def related_work(b: Builder):
    b.h1("3  Related work")
    b.p("A literature search (9 November 2026, about twelve searches) looked for neural networks that treat a "
        "galaxy as a cloud of its own particles and for contrastive models in astronomy.")
    b.bullets([
        "<b>Galaxies as clouds of their own particles</b> are rare. Nguyen, Mishra-Sharma, Williams &amp; Necib "
        "(2023, arXiv:2208.12825) apply a graph network to about 100 stars per mock dwarf galaxy (positions and "
        "line-of-sight velocities). Philipp Denzel's group at ZHAW used AdaPoinTr on TNG50 to predict gas points from "
        "dark-matter points, with at most about 10,000 input and 16,000 output points, and called phase-space point "
        "models “barely computationally tractable” (SKACH meeting talk, 2024).",
        "<b>Each galaxy as one point</b> is more common: Cuesta-Lazaro &amp; Mishra-Sharma (2023, arXiv:2311.17141) "
        "with 5,000 halos per simulation box, NeHOD (Nguyen et al. 2024, arXiv:2409.02980) with about 100 satellites "
        "per host, and Larson, Wu &amp; Jones (2024, arXiv:2407.13735) with TNG50 galaxies linked within 3 Mpc.",
        "<b>Contrastive models in astronomy</b> align other modalities: AstroCLIP (images and spectra) and AstroM3 "
        "(time series, spectra and images). No CLIP-style model on particle point clouds was found; not finding one "
        "does not prove that none exists.",
    ])
    b.p("With 2,048 points per modality, four modalities and about five thousand galaxies, this project works at "
        "the scale between the two galaxy-as-cloud studies above, and it is, as far as the search shows, the first "
        "to align several physical components of a galaxy contrastively.")


def data_section(b: Builder):
    n = b.n
    d, f = n["dataset"], n["facts"]
    st, raw = f["storage"], f["raw_snapshot"]
    sp = d["splits"]
    b.h1("4  Data")
    b.figure("pipeline.png", "From the raw simulation to training samples. Each step is a script in the "
             "repository; its counts are recorded in the dataset's files.")
    b.h2("4.1  Raw simulation data")
    b.p(f"The point clouds are built from the particle data of TNG50-1 at snapshot 99. An existing copy on the "
        f"cluster turned out to be incomplete: an audit found {raw['old_copy_missing_files']} files missing, so that "
        f"only {raw['old_copy_complete_2d_halos']} halos with 2D maps had all their particles. A fresh copy was "
        f"therefore downloaded with a resumable downloader written for the purpose ({raw['size_tb']} TB in "
        f"{raw['files']} snapshot files plus group catalogs; {raw['download']}; {raw['refused_retried']} refused "
        f"requests were retried and none failed). All files were checked against the server's sizes and a sample "
        f"against TNG's checksums. A second snapshot (91) was started for a multi-epoch dataset but stopped at 29% "
        f"when the research group's 10 TiB storage quota filled; it was deleted again.")
    b.h2("4.2  Galaxy selection")
    b.p("Each object is one Subfind subhalo, centrals and satellites alike, with the particles bound to it. Three "
        "filters apply (Table 1); they reproduce the selection of the GalaxyCLIP-2D maps.")
    b.table("Galaxy selection at snapshot 99.", ["Filter", "Reason"], [
        ["SubhaloFlag = 1", "real galaxies, not clumps formed inside another galaxy"],
        ["≥ 4,096 bound star and dark-matter particles", "twice the 2,048-point budget, so every epoch can draw "
                                                         "different points"],
        ["stellar half-mass radius ≥ 0.576 kpc", "two gravitational softening lengths; smaller galaxies have no "
                                                 "resolved shape"],
    ], widths=[0.42, 0.58], align_right_from=9)
    b.p(f"This selects <b>{d['selected']:,} galaxies</b> ({d['selected_centrals']:,} centrals and "
        f"{d['selected_satellites']:,} satellites) in {d['halos']:,} halos. The loader additionally requires 4,096 "
        f"star and dark-matter particles inside the sphere it uses; at the default sphere of 5 R½, "
        f"<b>{d['usable_r5']:,} galaxies</b> qualify, with stellar masses from 10<super>"
        f"{d['log_mstar_usable']['min']:.1f}</super> to 10<super>{d['log_mstar_usable']['max']:.1f}</super> solar "
        f"masses (median 10<super>{d['log_mstar_usable']['median']:.1f}</super>).")
    b.h2("4.3  Particles and storage")
    b.p(f"For every galaxy the bound particles within 10 R½ are stored: positions relative to the galaxy centre "
        f"(with the periodic box wrapped) in kpc, velocities relative to the galaxy's bulk motion in km/s, and the "
        f"physical fields of each particle type. Very large galaxies have millions of particles, so at most about "
        f"{st['cap']:,} (2<super>18</super>) particles per type are kept. The thinning is random with an inclusion "
        f"probability proportional to each particle's sampling weight, and a correction weight 1/probability keeps "
        f"later draws unbiased. The cap applies to {st['capped_galaxies']} galaxies; measured on 60 of them read in "
        f"full, it adds at most about 1% noise at 5 R½ ({st['cap_cost']}). In total "
        f"{st['particles_billion']} billion particles are stored (stars {st['stars_m']} million, dark matter "
        f"{st['dm_m']} million, gas {st['gas_m']} million) in 83 files of {st['shards_gb']} GB.")
    b.h2("4.4  The four modalities")
    b.p("The loader draws 2,048 points per modality from the particles inside the sphere (Table 2). Positions are "
        "divided by the sphere radius and velocities by the galaxy's maximum circular velocity, so absolute size "
        "and mass are not given to the model directly; the physical values are standardised. Gas and HI points "
        "are placed at random inside their cell's volume, so the two clouds never share coordinates.")
    b.table("What each modality contains. Features: x position, v velocity, Z metallicity, n<sub>H</sub> hydrogen "
            "density, T temperature, B magnetic field, x<sub>HI</sub> neutral fraction, SF star-forming flag.",
            ["Modality", "Drawn from", "Weighted by", "Features per point"], [
                ["stars", "star particles", "mass, without replacement", "x, v, age, Z"],
                ["dark matter", "dark-matter particles", "uniform, without replacement", "x, v"],
                ["gas", "gas cells", "mass, without replacement", "x, v, n<sub>H</sub>, T, Z, B, x<sub>HI</sub>, SF"],
                ["HI", "gas cells", "neutral-hydrogen mass, with replacement", "x, v, x<sub>HI</sub>"],
            ], widths=[0.15, 0.22, 0.31, 0.32], align_right_from=9)
    b.p(f"Not every galaxy has enough gas. The gas cloud is switched on when the sphere holds at least 4,096 gas "
        f"cells, the HI cloud when the effective number of HI-weighted cells reaches 512. Among the "
        f"{d['usable_r5']:,} usable galaxies, {d['gas_off_r5']:,} ({100 * d['gas_off_r5'] / d['usable_r5']:.0f}%) "
        f"have no gas cloud and {d['hi_off_r5']:,} ({100 * d['hi_off_r5'] / d['usable_r5']:.0f}%) no HI cloud, "
        f"mostly small galaxies and satellites whose gas was stripped by their host (Figure 3b). They stay in the "
        f"dataset with those modalities masked; the loss and the metrics skip masked pairs.")
    b.figure("dataset.png", "The usable galaxies at 5 R½. (a) Stellar-mass distribution of the three splits. "
             "(b) Share of galaxies with enough gas for a gas or HI cloud, by stellar mass. (c) Particles inside "
             "the sphere per galaxy: every usable galaxy has at least 4,096 star and dark-matter particles, twice "
             "the 2,048 points drawn.")
    b.h2("4.5  Splits")
    b.p(f"Galaxies in one halo share their environment and can look alike, so the split is done by halo: whole "
        f"halos go to training, validation or test (80/10/10 of the galaxies, balanced over stellar mass, seed "
        f"{d['settings']['seed']}). For later snapshots the split unit is the z = 0 halo in which a galaxy's "
        f"SubLink main branch ends, fetched from the TNG web API, so a galaxy and its own progenitor can never "
        f"land in different splits. Validation and test use frozen clouds, one fixed draw and orientation per "
        f"galaxy, so every run is scored on the same input (Table 3).")
    b.table("Usable galaxies at 5 R½ per split.", ["Split", "Galaxies", "Centrals", "With gas cloud",
                                                   "With HI cloud", "Halo has 2D maps"],
            [[s, f"{sp[s]['galaxies']:,}", f"{sp[s]['centrals']:,}", f"{sp[s]['gas_on']:,}", f"{sp[s]['hi_on']:,}",
              f"{sp[s]['halo_has_2d']:,}"] for s in ("train", "val", "test")],
            widths=[0.2, 0.16, 0.16, 0.16, 0.16, 0.16])
    b.h2("4.6  Quality checks")
    b.p("The pipeline is tested end to end on two synthetic snapshots, including adding the second snapshot to a "
        "dataset built from the first. Two builds of the real dataset are identical, and a quality report draws "
        "the clouds of twelve galaxies across stellar mass. The current version (v3) has the same snapshot-99 "
        "content as its predecessor, checked array by array, but a layout that can hold several snapshots.")


def method_section(b: Builder):
    n = b.n
    c = n["final_config"]
    tr = n["final_train"]
    b.h1("5  Method")
    b.figure("model.png", "The model. Each modality has its own PointNet encoder; a galaxy's four clouds are "
             "mapped to unit vectors in one 512-dimensional space and trained to lie close together. Faint points: "
             "other galaxies.")
    b.h2("5.1  Encoders")
    b.p(f"Each modality has its own PointNet: a per-point network with layer widths "
        f"{'–'.join(str(w) for w in c['pointnet_widths'])}, a maximum over the points, and a projection to a "
        f"{c['embed_dim']}-dimensional vector that is normalised to unit length. The four encoders of the final "
        f"setup have {n['facts']['compute']['params_m']} million parameters together. A patch transformer in the "
        f"style of Point-MAE ({n['facts']['phase_b_patch_params_m']} million parameters per modality) was tested "
        f"as an alternative (Section 6.3).")
    b.h2("5.2  Contrastive loss")
    b.p("For a pair of modalities a and b and a batch of B galaxies that have both, the similarity of galaxy i in "
        "modality a and galaxy j in modality b is the dot product of their embeddings divided by the temperature. "
        "The symmetric InfoNCE loss treats each row and each column of this similarity matrix as a classification "
        "of the true partner:")
    b.equation(r"\mathcal{L}_{ab} = -\frac{1}{2B}\sum_{i=1}^{B}\left[\log\frac{e^{s_{ii}}}{\sum_{j} e^{s_{ij}}}"
               r"+\log\frac{e^{s_{ii}}}{\sum_{j} e^{s_{ji}}}\right],\qquad s_{ij}=\frac{\mathbf{z}^{a}_{i}\cdot"
               r"\mathbf{z}^{b}_{j}}{\tau}", "infonce")
    b.p(f"The training loss is the mean of L<sub>ab</sub> over the six modality pairs. A pair is computed only on "
        f"the galaxies of the batch that have both modalities, so galaxies without gas still train the stars–dark "
        f"matter pair. The temperature τ starts at {c['temperature']} and is learned; in the final run it ends at "
        f"{tr['final_temperature']:.3f}.")
    b.h2("5.3  Orientation")
    b.p("A galaxy's clouds come from the same object, so in the simulation's frame they share an orientation: the "
        "stars and the gas rotate in the same plane. A model can match clouds by orientation alone, which says "
        "nothing about their physics. Every cloud therefore gets its own random rotation, and a mirror image half "
        "of the time, applied to positions and velocities together. Section 6.2 measures how large the shortcut "
        "is.")
    b.h2("5.4  Training")
    b.p(f"Batches of {c['batch_size']} galaxies are drawn evenly across six stellar-mass bins. The optimiser is "
        f"AdamW (learning rate {c['lr']:g}, weight decay {c['weight_decay']}), with {c['warmup_epochs']} warm-up "
        f"epochs followed by a cosine decay to zero over {c['epochs']} epochs, bfloat16 mixed precision and "
        f"adaptive gradient clipping (ZClip). Every epoch draws new points and new rotations. One run takes "
        f"{n['facts']['compute']['hours_per_run']} hours on one H100 GPU and peaks at "
        f"{tr['gpu_peak_gb']:.1f} GB of GPU memory.")
    b.h2("5.5  Evaluation")
    b.p("Every five epochs the model embeds the validation galaxies, and each galaxy is searched for in every other "
        "modality among all galaxies that have it (galleries of 520 galaxies for dark matter, 394 for gas and 472 "
        "for HI on validation; 499, 347 and 422 on test). A search counts as a hit only if the true partner is "
        "strictly the most similar; ties count against. The headline score is R@1 averaged over the 10 search "
        "directions that do not pair gas with HI, because those two clouds are drawn from the same gas cells and "
        "match almost trivially. The checkpoint with the best validation score is kept. The test set was scored "
        "once, after the final setup had been fixed on validation, and intervals come from bootstrapping the test "
        "galaxies.")
    b.h2("5.6  Linear probes")
    b.p("To see what the embeddings contain, ridge regressions predict 18 galaxy properties (masses, star-formation "
        "rate, metallicities, colour, stellar age, velocity dispersion, spin, size and maximum circular velocity) and "
        "two yes/no properties (central, quenched) from each modality's frozen embedding. Each probe is fitted on "
        "the training galaxies, its regularisation is chosen on validation, and it is scored on test. Two "
        "references use the same probes: about 20 hand-made, orientation-free statistics per cloud, and the same "
        "network with its untrained random weights.")


def results_section(b: Builder):
    n = b.n
    t = n["test"]
    bv = n["facts"]["baselines_val"]
    bt = n["baselines_test"]
    b.h1("6  Results")
    b.h2("6.1  Baselines without learning")
    b.p("Two baselines set the floor. The first matches clouds by the direction of their angular momentum, the "
        "second by about 20 orientation-free statistics per cloud, aligned across modalities with canonical "
        "correlation analysis (CCA) fitted on the training set (Table 4).")
    b.table("Baselines without a network: R@1 (%), mean over directions without and with gas–HI.",
            ["Baseline", "val, 10 dir.", "val, 12 dir.", "test, 10 dir.", "test, 12 dir."], [
                ["spin direction, clouds in a shared frame", f1(bv["spin_shared"][1]), f1(bv["spin_shared"][0]),
                 f1(bt["orientation: spin direction, shared frame"]["r1_10dir"]),
                 f1(bt["orientation: spin direction, shared frame"]["r1_12dir"])],
                ["spin direction, independent rotations", f1(bv["spin_independent"][1]),
                 f1(bv["spin_independent"][0]),
                 f1(bt["orientation: spin direction, independent rotations"]["r1_10dir"]),
                 f1(bt["orientation: spin direction, independent rotations"]["r1_12dir"])],
                ["statistics + CCA", f1(bv["stats_cca"][1]), f1(bv["stats_cca"][0]),
                 f1(bt["statistics + CCA"]["r1_10dir"]), f1(bt["statistics + CCA"]["r1_12dir"])],
            ], widths=[0.44, 0.14, 0.14, 0.14, 0.14],
            note="Chance: 0.23% on validation, 0.25% on test. 10 dir.: the 10 search directions without gas–HI.")
    b.p("Without any learning, the spin direction alone finds the right galaxy for about one in five queries when "
        "the clouds share their orientation, and for almost none once they are rotated independently. The "
        "statistics baseline reaches 4% without gas–HI; with gas–HI it reaches 15%, because two clouds drawn from "
        "the same gas cells share most of their statistics.")
    b.h2("6.2  The orientation shortcut")
    b.p("To measure the shortcut with a learned model, the first PointNet was also trained on clouds that share one "
        "rotation per galaxy (Table 5). Scored on such aligned clouds, it reaches 75.4% test R@1, twice the final "
        "model. On independently rotated clouds it collapses to 6.0%, barely above the statistics baseline: it had "
        "learned to compare orientations, not galaxies. With independent rotations nothing of this kind can be "
        "learned, and a model trained that way scores the same on aligned and independent clouds.")
    b.md_table("orientation_shortcut", widths=[0.4, 0.15, 0.15, 0.15, 0.15])
    b.h2("6.3  First models: PointNet and patch transformer")
    abl = {r[0].replace("&nbsp;", "").strip(): r for r in n["tables"]["phase_c_ablations"]["rows"]}
    pb, pt = abl["Phase B PointNet"], abl["Phase B patch transformer (reference)"]
    b.p(f"The first comparison (Phase B) used one PointNet per modality (1.2 million parameters each, 300 epochs) "
        f"and a patch transformer (11.9 million each, 500 epochs). The PointNet reached {pb[1]}% validation R@1 (best epoch "
        f"{pb[4]}). The patch transformer peaked at {pt[1]}% at epoch {pt[4]} and then fell to {pt[5]}% by the end "
        f"of training: with about four thousand training galaxies it memorises them. Smaller transformers, dropout "
        f"and radially spread patch centres reduced but did not remove the overfitting (best 27.8%), so the PointNet "
        f"was kept.")
    b.figure("thesis_fig1_training_curves.png", "Validation R@1 (10 directions, 520 galaxies) during training, "
             "every 5 epochs. Blue: the final setup, three seeds. Orange: the Phase B PointNet (300 epochs). Green: "
             "the control trained on clouds that share one rotation per galaxy, scored on independently rotated "
             "clouds. Dashed: statistics + CCA, no network.")
    b.h2("6.4  Ablations")
    b.p("Phase C changed one setting at a time against the Phase B PointNet (Figure 6, Table 6). A change counted "
        "only if it moved the best validation score by more than 2 points, a threshold set before the runs (two "
        "seeds of the reference differ by 0.3).")
    b.bullets([
        "<b>Helped, each just below the threshold, and they add up:</b> a learned temperature (+1.7), a wider "
        "encoder (+1.4) and 600 instead of 300 epochs (+1.3). Combined, over three seeds, they give 38.0% (+4.8). "
        "This combination is the final setup.",
        "<b>Helped alone, redundant in combination:</b> rotation-free point features (distance, speed, radial and "
        "tangential velocity) gave the largest single gain (+3.5), but added to the combination they changed "
        "nothing (37.7% over three seeds).",
        "<b>Hurt:</b> batch 512 (−3.7), a narrow encoder (−3.6), one encoder shared by all modalities (−4.2) and "
        "1,024 points per cloud (−3.2). 4,096 points helped a little (+1.0).",
        "<b>Inputs that matter:</b> without velocities the score drops by 11.7 points and the stars–dark matter pair "
        "almost vanishes (1.7%); without the per-point physical values training never starts.",
    ])
    b.figure("thesis_fig2_phase_c_ablations.png", "Phase C ablations: best validation R@1 for one change at a time. "
             "Shaded band: the Phase B PointNet ± 2 points, the threshold set before the runs.")
    b.md_table("phase_c_ablations", widths=[0.40, 0.14, 0.11, 0.11, 0.12, 0.12], size=7.3, long=True)
    b.h2("6.5  The final model on the test set")
    b.p(f"The final setup (learned temperature, wide PointNet, 600 epochs) was trained with three seeds and scored "
        f"once on the 499 test galaxies (Table 7). It reaches <b>{f1(t['r1_final'])}%</b> R@1 over the 10 "
        f"directions (seed spread ± 0.2, 95% interval {f1(t['interval'][0])}–{f1(t['interval'][1])}), "
        f"{f1(t['final_minus_base'])} points above the Phase B PointNet (interval {f1(t['diff_interval'][0])}–"
        f"{f1(t['diff_interval'][1])}). The test score is about two points above validation; with about five "
        f"hundred galaxies per set the sampling uncertainty alone is ±1.6 points, so the two agree. With gas–HI the "
        f"mean is 49.2%, and R@10 is 74.9%.")
    b.md_table("retrieval_main", widths=[0.30, 0.10, 0.10, 0.12, 0.10, 0.13, 0.15], size=7.2)
    b.p("Per pair (Table 8, Figure 7), stars–gas (87.6%) and stars–HI (63.4%) are matched best; the dark matter is "
        "the hardest modality (stars–dark matter 24.1%, dark matter–gas 16.2%, dark matter–HI 9.8%). These are still "
        "50 to 100 times chance, and the gains over the first PointNet are largest exactly there: stars–dark matter "
        "nearly doubles from 12.8% to 24.1%.")
    b.md_table("retrieval_pairs", widths=[0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125], size=7.2)
    b.figure("thesis_fig3_test_retrieval_pairs.png", "Test R@1 per modality pair, mean of both search "
             "directions. Whiskers: 95% bootstrap intervals over the 499 test galaxies. Black tick: statistics + "
             "CCA without a network.")
    b.h2("6.6  What the embeddings know")
    pm = {r[0]: r for r in n["tables"]["probes_summary"]["rows"]}
    b.p(f"Linear probes on the frozen embeddings (Table 9, Figure 8) recover the 18 galaxy properties with a mean "
        f"test R² of {pm['final setup (3 seeds)'][1].split(' ')[0]} from the stars, "
        f"{pm['final setup (3 seeds)'][3].split(' ')[0]} from the gas, {pm['final setup (3 seeds)'][4].split(' ')[0]} "
        f"from the HI and {pm['final setup (3 seeds)'][2].split(' ')[0]} from the dark matter. Hand-made "
        f"statistics of the same clouds reach {pm['hand-made statistics'][1].split(' ')[0]}, "
        f"{pm['hand-made statistics'][3].split(' ')[0]}, {pm['hand-made statistics'][4].split(' ')[0]} and "
        f"{pm['hand-made statistics'][2].split(' ')[0]}. The largest gains are for the HI and the dark matter: the "
        f"training signal from the other modalities teaches these encoders what matters about the galaxy. The "
        f"dark-matter embedding alone predicts stellar mass with R² 0.80, black-hole mass with 0.79 and stellar "
        f"metallicity with 0.78; statistics of the same cloud reach 0.29, 0.39 and 0.30 (Figure 9).")
    b.md_table("probes_summary", widths=[0.28, 0.18, 0.18, 0.18, 0.18], size=7.4)
    b.figure("thesis_fig4_probe_means.png", "Linear probes on frozen embeddings: mean test R² over 18 galaxy "
             "properties for each modality's embedding (negative R² counted as 0).", width=0.86 * TEXT_W)
    b.figure("thesis_fig5_probe_properties.png", "Test R² per property for the final setup (a) and its gain over "
             "hand-made statistics of the same clouds (b). Each column uses only that modality's cloud.",
             width=0.9 * TEXT_W)
    dm_control(b)
    matched_control(b)


def dm_control(b: Builder):
    n = b.n
    dm, sub = n["dm_control"], n["rvmax_subset"]
    r = dm["retrieval"]
    pr = dm["probes"]
    b.h2("6.7  Control 1: is the dark-matter result real?")
    b.p("Every cloud, the dark matter included, is cut at 5 R½ and scaled by that radius. The dark-matter cloud "
        "therefore carries the stars' size in its own shape, and the size of the stellar body says a lot about "
        "stellar mass. The dark-matter probes of Section 6.6 could partly read this imprint instead of the dark "
        "matter's own structure.")
    b.p(f"The control (run c29) cuts the dark matter at the halo's own radius R<sub>vmax</sub> and scales it by "
        f"R<sub>vmax</sub>; stars, gas and HI keep the stellar sphere, and all other settings are those of the final "
        f"setup (one seed). The R<sub>vmax</sub> sphere must lie inside the stored 10 R½ and hold at least 4,096 "
        f"dark-matter particles, which keeps {sub['kept']:,} of the {sub['usable']:,} usable galaxies "
        f"({100 * sub['kept'] / sub['usable']:.0f}%; {sub['beyond_extent']} have R<sub>vmax</sub> beyond the stored "
        f"particles, {sub['too_few_dm']} too few particles inside it): {sub['splits']['train']:,} training and "
        f"{sub['splits']['val']} validation galaxies. All comparisons below use exactly these validation galaxies "
        f"and their training counterparts; the test set is not touched.")
    props = [("log_mstar_total", "stellar mass"), ("rhalf_star_kpc", "stellar half-mass radius"),
             ("gas_fraction", "gas fraction"), ("log_mdm_total", "dark-matter mass"), ("log_mbh", "black-hole mass"),
             ("log_z_stars", "stellar metallicity"), ("vmax", "maximum circular velocity")]
    rows = [[lab] + [f2(pr[k]["dm"]["props"][p]) for k in ("C21-stellar", "c29-Rvmax", "stats-stellar",
                                                             "stats-Rvmax")] for p, lab in props]
    rows.append(["mean of 18 properties"] + [f2(pr[k]["dm"]["mean_r2"]) for k in ("C21-stellar", "c29-Rvmax",
                                                                                    "stats-stellar", "stats-Rvmax")])
    pairs = [("stars-dm", "stars – dark matter"), ("dm-gas", "dark matter – gas"), ("dm-hi", "dark matter – HI"),
             ("stars-gas", "stars – gas"), ("stars-hi", "stars – HI")]
    rrows = [[lab, f1(r["C21-stellar"]["pairs"][p]), f1(r["c29-Rvmax"]["pairs"][p])] for p, lab in pairs]
    rrows.append(["mean, 10 directions", f1(r["C21-stellar"]["r1_no_gas_hi"]), f1(r["c29-Rvmax"]["r1_no_gas_hi"])])
    b.dm_table = b.table(f"Retrieval on the same {dm['shared']['val']} validation galaxies, R@1 (%), mean of both "
                         f"directions.", ["Pair", "C21, stellar frame", "c29, halo frame"], rrows,
                         widths=[0.5, 0.25, 0.25])
    b.table("What the dark-matter embedding predicts: validation R² on the same galaxies. Statistics: hand-made "
            "statistics of the dark-matter cloud in each frame.",
            ["Property", "C21, stellar frame", "c29, halo frame", "statistics, stellar frame",
             "statistics, halo frame"], rows, widths=[0.32, 0.17, 0.17, 0.17, 0.17], bold_rows=(len(rows) - 1,))
    if n.get("c30"):
        b.p(f"Figures {b.fig_no + 1} and {b.fig_no + 2} in Section 6.8 show these numbers next to the matched control.")
    else:
        control_figures(b)
    b.p(f"<b>The imprint was real.</b> The clearest sign is the galaxy's own size: the final setup's dark-matter "
        f"embedding predicted R½ with R² {f2(pr['C21-stellar']['dm']['props']['rhalf_star_kpc'])}, the halo-frame "
        f"one only with {f2(pr['c29-Rvmax']['dm']['props']['rhalf_star_kpc'])}. Stars–dark matter retrieval halves "
        f"({f1(r['C21-stellar']['pairs']['stars-dm'])}% → {f1(r['c29-Rvmax']['pairs']['stars-dm'])}%), and dark "
        f"matter–gas drops from {f1(r['C21-stellar']['pairs']['dm-gas'])}% to {f1(r['c29-Rvmax']['pairs']['dm-gas'])}%; "
        f"pairs without dark matter are unchanged, as their inputs are.")
    b.p(f"<b>Size-linked properties drop, halo-linked ones hold.</b> Stellar mass falls from "
        f"{f2(pr['C21-stellar']['dm']['props']['log_mstar_total'])} to "
        f"{f2(pr['c29-Rvmax']['dm']['props']['log_mstar_total'])}, and the gas fraction falls too, while "
        f"dark-matter mass, black-hole mass, stellar metallicity and maximum circular velocity stay where they "
        f"were.")
    b.p(f"<b>The network still learns real dark-matter structure.</b> In the halo frame the trained embedding "
        f"beats statistics of the same clouds (mean R² {f2(pr['c29-Rvmax']['dm']['mean_r2'])} against "
        f"{f2(pr['stats-Rvmax']['dm']['mean_r2'])}), and stars–dark matter R@1 of "
        f"{f1(r['c29-Rvmax']['pairs']['stars-dm'])}% is {r['c29-Rvmax']['pairs']['stars-dm'] * dm['shared']['val'] / 100:.0f} "
        f"times chance. Note that in the halo frame simple "
        f"statistics already predict stellar mass well ({f2(pr['stats-Rvmax']['dm']['props']['log_mstar_total'])}), "
        f"because the number of particles inside R<sub>vmax</sub> measures the halo's mass.")
    b.p("<b>Caveats.</b> One seed (seed-to-seed differences of the final setup are far smaller: stars–dark matter "
        "20.1 vs 22.6, dark-matter stellar-mass R² 0.80 vs 0.78). R<sub>vmax</sub> comes from the total mass profile, "
        "stars included, so the halo frame removes the stars' size but not every trace of the baryons. And the "
        "control trained on 80% of the galaxies, "
        + ("which, as the matched control in Section 6.8 shows, is not what lowered its numbers." if n.get("c30")
           else "which is what the matched control tests."))


def control_figures(b: Builder):
    """The two figures of the dark-matter controls: in Section 6.7 while the matched control is missing, in Section
    6.8 (all three runs) once it exists."""
    n = b.n
    three = bool(n.get("c30"))
    runs = ("the halo-frame control (c29), the final setup (C21) and the matched control (c30)" if three
            else "the final setup (C21) and the halo-frame control (c29)")
    tables = f"Tables {b.dm_table} and {b.tab_no}" if three else f"Table {b.dm_table}"
    b.figure("dm_control.png", f"Dark-matter controls on the same {n['dm_control']['shared']['val']} validation "
             f"galaxies: {runs}. (a) Retrieval per pair. (b) Validation R² of linear probes on the dark-matter "
             f"embedding; lines mark hand-made statistics of the dark-matter cloud in the stellar frame (solid) and "
             f"the halo frame (dashed).")
    b.figure("control_curves.png", f"Validation R@1 during training for {runs}. The final setup is scored on its "
             f"{n['dataset']['splits']['val']['galaxies']} validation galaxies, the controls on their "
             f"{n['dm_control']['shared']['val']}, so the curves show the course of training; the like-for-like "
             f"comparison is in {tables}.")


def matched_control(b: Builder):
    n = b.n
    c30, dm = n.get("c30"), n["dm_control"]
    b.h2("6.8  Control 2: the same galaxies in the stellar frame")
    b.p("The halo-frame run differs from the final setup in two ways: the dark-matter frame and the training set "
        "(the 80% of galaxies for which an R<sub>vmax</sub> sphere can be drawn). The matched control (run c30) "
        "trains the final setup in the usual stellar frame on exactly the halo-frame run's galaxies. Its "
        "validation clouds are the same frozen clouds the final setup was scored on, so it differs from the final "
        "setup only in the training galaxies and from the halo-frame run only in the dark-matter frame.")
    if not c30:
        b.p("<i>This run was still waiting for a GPU when this version of the report was built. Its result will be "
            "added here.</i>")
        return
    r, pr = c30["retrieval"], c30["probes"]
    keys = [k for k in ("C21-stellar", "c30-subset", "c29-Rvmax") if k in r]
    names = {"C21-stellar": "C21, stellar frame, all galaxies", "c30-subset": "c30, stellar frame, same galaxies",
             "c29-Rvmax": "c29, halo frame"}
    rows = [["stars – dark matter R@1 (%)"] + [f1(r[k]["pairs"]["stars-dm"]) for k in keys],
            ["dark matter – gas R@1 (%)"] + [f1(r[k]["pairs"]["dm-gas"]) for k in keys],
            ["mean R@1, 10 directions (%)"] + [f1(r[k]["r1_no_gas_hi"]) for k in keys],
            ["DM embedding: stellar mass R²"] + [f2(pr[k]["dm"]["props"]["log_mstar_total"]) for k in keys],
            ["DM embedding: half-mass radius R²"] + [f2(pr[k]["dm"]["props"]["rhalf_star_kpc"]) for k in keys],
            ["DM embedding: mean R² of 18"] + [f2(pr[k]["dm"]["mean_r2"]) for k in keys],
            ["stars embedding: mean R² of 18"] + [f2(pr[k]["stars"]["mean_r2"]) for k in keys]]
    b.table(f"The three runs on the same {c30['shared']['val']} validation galaxies.", ["Score"] +
            [names[k] for k in keys], rows, widths=[0.34] + [0.66 / len(keys)] * len(keys))
    reading = n.get("c30_reading", "")
    for para in reading if isinstance(reading, list) else [reading]:
        b.p(para)
    control_figures(b)


def discussion(b: Builder):
    c30 = b.n.get("c30")
    fewer = (c30["retrieval"]["C21-stellar"]["r1_no_gas_hi"] - c30["retrieval"]["c30-subset"]["r1_no_gas_hi"]
             if c30 else 0.0)
    b.h1("7  Discussion and limitations")
    b.p("The main result is that contrastive learning on raw particle point clouds works: from 2,048 points per "
        "component, the model identifies the matching component of the same galaxy among about five hundred "
        "candidates for 40% of queries, ten times better than hand-made statistics, and its embeddings carry the "
        "galaxy's physical properties. The dark matter is both the hardest modality to match and the one whose "
        "embedding gains most from the other modalities' supervision.")
    b.p("Two findings concern the method more than the model. First, the orientation shortcut: with shared "
        "orientations a model scores almost twice the final result while learning nothing useful, so independent "
        "rotations are essential and any future modality (including 2D maps) must be checked the same way. Second, "
        "the scaling of the clouds is itself information: scaling every cloud by the stars' size leaks that size "
        "into the dark-matter cloud and inflates part of the dark-matter results"
        + ("; the matched control shows that this comes from the frame, not from the training set (Section 6.8)."
           if b.n.get("c30") else "."))
    b.h3("Limitations")
    b.bullets([
        "<b>One snapshot.</b> All galaxies are at z = 0; the dataset can hold more snapshots, but each needs 2.93 TB "
        "of raw data, more than the free storage, before it can be reduced to about 47 GB of point clouds.",
        "<b>Data size.</b> About five thousand galaxies are few for representation learning: the patch transformer "
        "overfits, and the PointNet's best epoch comes well before the end for some settings."
        + (f" For the PointNet itself, 20% fewer training galaxies cost only {fewer:.1f} points of mean R@1 "
           f"(Section 6.8)." if c30 else ""),
        "<b>Points per cloud.</b> 2,048 points sample galaxies that have up to millions of particles; 4,096 points "
        "helped slightly, at twice the cost.",
        "<b>Only bound particles</b> are used; gas around a galaxy that is not bound to it is not in the clouds.",
        "<b>Scaling choices.</b> Per-galaxy scaling removes absolute size by design, but the shared stellar radius "
        "leaks the stars' size into the dark matter (" + ("Sections 6.7 and 6.8)." if b.n.get("c30") else
                                                          "Section 6.7)."),
        "<b>Controls with one seed</b>, and <b>the test set has now been used</b>: further choices must be made on "
        "validation. A later snapshot cannot serve as a fresh test set, because it contains the same galaxies at "
        "another time.",
    ])


def next_steps(b: Builder):
    b.h1("8  Next steps")
    b.bullets([
        "<b>2D↔3D alignment.</b> Keep the trained 2D map model frozen, export its embeddings keyed by snapshot, "
        "halo and view, and train the 3D encoders to match them (anchor training as in ImageBind). Both datasets "
        "carry the halo identifiers this needs; the two projects' splits disagree, so a common split is required "
        "first.",
        b.n.get("c30_next_step") if b.n.get("c30") else
        "<b>Decide the dark-matter frame for the thesis.</b> With the matched control, choose whether to report the "
        "stellar-frame numbers with the halo-frame numbers beside them, or to switch the dark matter to the halo "
        "frame.",
        "<b>More snapshots.</b> Adding snapshot 91 (and later others) when storage allows would give more training "
        "galaxies and allow pairing galaxies with their progenitors.",
        "<b>More seeds for the controls</b>, and the 2D orientation check: test whether the 2D map model relies on "
        "shared orientations as the first 3D model did.",
        "<b>Writing.</b> The thesis figures and tables are generated from the saved results (tools/clip_report.py), "
        "so the 3D chapter can be written now.",
    ])


def appendix(b: Builder):
    n = b.n
    c = n["final_config"]
    b.add(PageBreak())
    b.h1("Appendix A  Hyperparameters and compute")
    rows = [["encoder", "PointNet per modality, widths " + "–".join(str(w) for w in c["pointnet_widths"])],
            ["parameters", f"{n['facts']['compute']['params_m']} million (four encoders)"],
            ["embedding", f"{c['embed_dim']}-dimensional, unit length"],
            ["loss", "symmetric InfoNCE, mean over the 6 modality pairs, masked pairs skipped"],
            ["temperature", f"learned, starts at {c['temperature']}, ends at {n['final_train']['final_temperature']:.3f}"],
            ["batch", f"{c['batch_size']} galaxies, balanced over six stellar-mass bins"],
            ["optimiser", f"AdamW, learning rate {c['lr']:g}, weight decay {c['weight_decay']}, betas 0.9 / 0.999"],
            ["schedule", f"{c['warmup_epochs']} warm-up epochs, cosine decay to 0, {c['epochs']} epochs"],
            ["precision, clipping", "bfloat16 mixed precision, ZClip adaptive gradient clipping (z = 2.5)"],
            ["input", "2,048 points per modality inside 5 R½, new draw and rotation every epoch"],
            ["orientation", "independent random rotation per cloud, mirror image half of the time"],
            ["evaluation", "every 5 epochs on 520 validation galaxies; best checkpoint by R@1 (10 directions)"],
            ["seeds", "4869, 2 and 3 (final setup); 4869 (controls)"],
            ["compute", f"one H100 GPU per run, {n['facts']['compute']['hours_per_run']} hours, "
                        f"{n['final_train']['gpu_peak_gb']:.1f} GB peak memory"]]
    b.table("Final setup.", ["Setting", "Value"], rows, widths=[0.25, 0.75], align_right_from=9)
    b.h1("Appendix B  Further tables")
    b.md_table("probes_auc", widths=[0.28, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12], size=7.2)
    b.md_table("probes_properties", widths=[0.24] + [0.095] * 8, size=7.0, long=True)
    b.h1("Appendix C  Code and reproducibility")
    b.p(f"All code is in the public repository <font color='{ACCENT}'>{REPO_URL}</font>. Results, checkpoints and "
        f"the dataset stay on the ZHAW cluster (/cluster/projects/ska/illustris/tng50-1.pointclouds/v3).")
    b.table("Where things are.", ["Task", "Code"], [
        ["select galaxies, merger-tree branches, splits", "tools/build_selection.py"],
        ["extract particles into shards", "slurm/build_pointclouds.submit → tools/build_pointclouds.py"],
        ["indexes, feature statistics, frozen validation and test clouds", "tools/finalize_pointclouds.py"],
        ["quality checks", "tools/check_pointclouds.py"],
        ["draw point clouds, PyTorch loader", "galaxyclip3d/pointclouds/sampling.py, galaxyclip3d/dataset.py"],
        ["masked all-pairs loss and full-gallery recall", "galaxyclip3d/multimodal.py"],
        ["encoders", "galaxyclip3d/models/"],
        ["training", "galaxyclip3d/train_clip.py, configs/clip_*.json, slurm/train_clip.submit"],
        ["baselines without a network", "tools/clip_baselines.py"],
        ["scoring, probes, comparisons", "tools/eval_clip.py, tools/probe_clip.py, tools/compare_probes.py"],
        ["thesis figures and tables", "tools/clip_report.py"],
        ["this report and the slides", "report/ (collect_numbers.py, figures.py, build_pdf.py, build_deck.js)"],
        ["tests", "tests/test_pointclouds.py, tests/test_clip.py"],
    ], widths=[0.48, 0.52], align_right_from=9, size=7.4)
    b.h1("Appendix D  Timeline")
    b.table("Project timeline (2026).", ["Date", "Step"], [[d_, s_] for d_, s_ in n["timeline"]],
            widths=[0.14, 0.86], align_right_from=9)


def build() -> None:
    n = json.loads((OUT / "numbers.json").read_text())
    b = Builder(n)
    title_page(b)
    contents(b)
    summary(b)
    introduction(b)
    background(b)
    related_work(b)
    data_section(b)
    method_section(b)
    results_section(b)
    discussion(b)
    next_steps(b)
    appendix(b)
    doc = Doc(PDF)
    doc.multiBuild(b.story)
    print(f"wrote {PDF}")


if __name__ == "__main__":
    build()
