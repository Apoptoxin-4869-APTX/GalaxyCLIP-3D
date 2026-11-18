"""Approximate slide images and a text-overflow check for the deck, for machines without PowerPoint or LibreOffice.

    cd GalaxyCLIP_3D && PYTHONPATH=~/tools/report_pylibs:pylibs:. python -s report/preview_deck.py
    # -> outputs/project_report/deck_preview/slide-NN.png and an overflow report on stdout

Text is measured with DejaVu fonts scaled to Calibri/Cambria widths (Calibri is about 12% narrower than DejaVu
Sans), so the check is approximate; charts are drawn from their data with matplotlib, not as PowerPoint would draw
them. It catches text that does not fit its box, elements off the slide and crowded layouts, not exact looks.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from PIL import Image, ImageDraw, ImageFont                              # noqa: E402
from pptx import Presentation                                            # noqa: E402
from pptx.enum.dml import MSO_THEME_COLOR                                # noqa: E402
from pptx.util import Emu                                                # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/project_report"
DECK = OUT / "GalaxyCLIP3D_supervisor_deck.pptx"
PREV = OUT / "deck_preview"
DPI = 110
DV = Path("/usr/share/fonts/truetype/dejavu")
THEME = {MSO_THEME_COLOR.TEXT_1: "0B1B33", MSO_THEME_COLOR.DARK_1: "0B1B33", MSO_THEME_COLOR.BACKGROUND_1: "FFFFFF",
         MSO_THEME_COLOR.LIGHT_1: "FFFFFF", MSO_THEME_COLOR.TEXT_2: "1F3A5F", MSO_THEME_COLOR.DARK_2: "1F3A5F",
         MSO_THEME_COLOR.BACKGROUND_2: "F3F2EE", MSO_THEME_COLOR.LIGHT_2: "F3F2EE", MSO_THEME_COLOR.ACCENT_1: "2A78D6",
         MSO_THEME_COLOR.ACCENT_2: "EB6834", MSO_THEME_COLOR.ACCENT_3: "1BAF7A", MSO_THEME_COLOR.ACCENT_4: "EDA100",
         MSO_THEME_COLOR.ACCENT_5: "6250D6", MSO_THEME_COLOR.ACCENT_6: "E34948"}
WIDTH_SCALE = {"Calibri": 0.88, "Cambria": 0.92}
_fonts: dict = {}


def font(face: str, bold: bool, size_px: float):
    serif = face == "Cambria"
    name = ("DejaVuSerif" if serif else "DejaVuSans") + ("-Bold" if bold else "") + ".ttf"
    key = (name, round(size_px * WIDTH_SCALE.get(face, 0.88), 1))
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(str(DV / name), max(1, int(round(key[1]))))
    return _fonts[key]


def px(emu) -> float:
    return Emu(emu).inches * DPI


def colour(c, default="0B1B33"):
    try:
        if c is None or c.type is None:
            return "#" + default
        if c.type == 1:                                          # RGB
            return "#" + str(c.rgb)
        return "#" + THEME.get(c.theme_color, default)
    except Exception:
        return "#" + default


def runs_of(par, default_size, default_face, default_bold, default_colour):
    out = []
    for r in par.runs:
        size = r.font.size.pt if r.font.size else default_size
        face = r.font.name or default_face
        bold = r.font.bold if r.font.bold is not None else default_bold
        col = colour(r.font.color, default_colour) if r.font.color and r.font.color.type is not None else "#" + default_colour
        out.append((r.text, size, face, bold, col))
    return out


def layout_text(frame, width_px, default_size=18, default_face="Calibri", default_bold=False, default_colour="0B1B33"):
    """Wrapped lines [(x_offset, y, [(text, font, colour)])] and the total height in px."""
    lines, y = [], 0.0
    for par in frame.paragraphs:
        runs = runs_of(par, default_size, default_face, default_bold, default_colour)
        bullet = par._p.pPr is not None and par._p.pPr.find(
            "{http://schemas.openxmlformats.org/drawingml/2006/main}buChar") is not None
        indent = 0.22 * DPI if bullet else 0
        if not runs or not "".join(r[0] for r in runs).strip():
            y += 1.2 * default_size / 72 * DPI
            continue
        size = max(r[1] for r in runs)
        lh = 1.18 * size / 72 * DPI
        words = []
        for text, s, face, bold, col in runs:
            for k, wd in enumerate(text.replace("\n", " \n ").split(" ")):
                words.append((wd, font(face, bold, s / 72 * DPI), col))
        cur, cur_w, first = [], 0.0, True
        for wd, f, col in words + [("\n", None, None)]:
            if wd == "\n":
                if cur:
                    lines.append((indent, y, cur, first and bullet))
                    y += lh
                    first = False
                cur, cur_w = [], 0.0
                continue
            ww = f.getlength(wd + " ")
            if cur and cur_w + ww > width_px - indent:
                lines.append((indent, y, cur, first and bullet))
                y += lh
                first = False
                cur, cur_w = [], 0.0
            cur.append((wd, f, col))
            cur_w += ww
        spc = par.space_after.pt if par.space_after is not None else 0
        y += spc / 72 * DPI
    return lines, y


def draw_lines(dr, lines, x, y, align="l", width=0):
    for dx, dy, words, mark in lines:
        line_w = sum(f.getlength(w + " ") for w, f, _ in words)
        ox = x + dx + ((width - line_w) / 2 if align == "c" else (width - line_w if align == "r" else 0))
        if mark:
            dr.text((x + dx - 0.15 * DPI, y + dy), "•", font=words[0][1], fill=words[0][2])
        for w, f, col in words:
            dr.text((ox, y + dy), w, font=f, fill=col)
            ox += f.getlength(w + " ")


def chart_image(ch, w, h):
    fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI)
    ax = fig.add_subplot(111)
    try:
        plot = ch.plots[0]
        cats = list(plot.categories)
        kind = ch.chart_type
        series = list(plot.series)
        n = len(series)
        horizontal = "BAR" in str(kind) and "COLUMN" not in str(kind)
        if "LINE" in str(kind):
            for s in series:
                ax.plot(range(len(cats)), list(s.values), lw=1.5)
        else:
            stacked = "STACKED" in str(kind)
            for i, s in enumerate(series):
                vals = [v or 0 for v in s.values]
                pos = [j + (0 if stacked else (i - (n - 1) / 2) * 0.8 / n) for j in range(len(cats))]
                (ax.barh if horizontal else ax.bar)(pos, vals, 0.8 / (1 if stacked else n))
            if horizontal:
                ax.set_yticks(range(len(cats)), cats, fontsize=7)
                ax.invert_yaxis()
            else:
                ax.set_xticks(range(len(cats)), cats, fontsize=7)
        ax.set_title(", ".join(s.name for s in series)[:90], fontsize=7)
    except Exception as e:                                                # noqa: BLE001
        ax.text(0.5, 0.5, f"chart ({e})", ha="center")
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    return Image.open(buf).convert("RGB").resize((int(w), int(h)))


def main() -> None:
    prs = Presentation(str(DECK))
    W, H = px(prs.slide_width), px(prs.slide_height)
    PREV.mkdir(parents=True, exist_ok=True)
    for f in PREV.glob("slide-*.png"):
        f.unlink()
    problems = []
    for si, slide in enumerate(prs.slides, start=1):
        dark = slide.slide_layout.name in ("TITLE", "SECTION")
        img = Image.new("RGB", (int(W), int(H)), "#0B1B33" if dark else "#FFFFFF")
        dr = ImageDraw.Draw(img)
        shapes = [s for s in slide.slide_layout.shapes if not s.is_placeholder] + list(slide.shapes)
        for sh in shapes:
            x, y, w, h = px(sh.left or 0), px(sh.top or 0), px(sh.width or 0), px(sh.height or 0)
            name = sh.name
            if x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1:
                problems.append(f"slide {si}: {name} extends beyond the slide")
            if sh.shape_type == 13:                                       # picture
                pic = Image.open(io.BytesIO(sh.image.blob)).convert("RGB").resize((max(1, int(w)), max(1, int(h))))
                img.paste(pic, (int(x), int(y)))
                continue
            if getattr(sh, "has_chart", False) and sh.has_chart:
                img.paste(chart_image(sh.chart, w, h), (int(x), int(y)))
                dr.rectangle([x, y, x + w, y + h], outline="#C3C2B7")
                continue
            if getattr(sh, "has_table", False) and sh.has_table:
                ty = y
                for r in sh.table.rows:
                    cx, rh = x, 0
                    cells = []
                    for ci, c in enumerate(r.cells):
                        cw = px(sh.table.columns[ci].width)
                        ls, th = layout_text(c.text_frame, cw - 0.12 * DPI, 12)
                        cells.append((cx, cw, ls))
                        rh = max(rh, th + 0.06 * DPI)
                        cx += cw
                    for cx, cw, ls in cells:
                        draw_lines(dr, ls, cx + 0.06 * DPI, ty + 0.03 * DPI)
                    ty += rh
                    dr.line([x, ty, x + w, ty], fill="#D9D7CF")
                if ty > H - 0.4 * DPI:
                    problems.append(f"slide {si}: table {name} runs to {ty / DPI:.2f} in (slide height {H / DPI:.2f})")
                continue
            # autoshapes: fill
            try:
                if sh.fill.type == 1:
                    fc = colour(sh.fill.fore_color, "F3F2EE")
                    geom = sh.auto_shape_type if hasattr(sh, "auto_shape_type") else None
                    if geom is not None and "OVAL" in str(geom):
                        dr.ellipse([x, y, x + w, y + h], fill=fc)
                    elif geom is not None and "TRIANGLE" in str(geom):
                        dr.polygon([(x, y), (x + w, y + h / 2), (x, y + h)], fill=fc)
                    else:
                        dr.rounded_rectangle([x, y, x + w, y + h], radius=int(0.08 * DPI), fill=fc)
            except Exception:                                             # noqa: BLE001
                pass
            if sh.has_text_frame and sh.text_frame.text.strip():
                ph_title = sh.is_placeholder and sh.placeholder_format.type is not None and "TITLE" in str(sh.placeholder_format.type)
                slide_no = sh.is_placeholder and "SLIDE_NUMBER" in str(sh.placeholder_format.type)
                size = 54 if (ph_title and dark and si == 1) else 44 if (ph_title and dark) else 34 if ph_title else \
                    10 if slide_no else 22 if sh.is_placeholder else 16
                face = "Cambria" if ph_title else "Calibri"
                col = "FFFFFF" if dark and sh.is_placeholder else "F3F2EE" if dark else "1F3A5F" if ph_title else "0B1B33"
                if sh.is_placeholder:                                      # the layout's placeholder box
                    for lp in slide.slide_layout.placeholders:
                        if lp.placeholder_format.idx == sh.placeholder_format.idx:
                            x, y, w, h = px(lp.left), px(lp.top), px(lp.width), px(lp.height)
                ls, th = layout_text(sh.text_frame, w, size, face, ph_title, col)
                align = sh.text_frame.paragraphs[0].alignment
                al = "c" if align is not None and "CENTER" in str(align) else "r" if align is not None and "RIGHT" in str(align) else "l"
                anchor = sh.text_frame._txBody.bodyPr.get("anchor")
                oy = y + (h - th) / 2 if anchor == "ctr" else y + h - th if anchor == "b" else y
                draw_lines(dr, ls, x, oy, al, w)
                if th > h + 0.04 * DPI:
                    problems.append(f"slide {si}: text overflows {name} ({th / DPI:.2f} in needed, {h / DPI:.2f} in box): "
                                    f"{sh.text_frame.text[:60]!r}")
        img.save(PREV / f"slide-{si:02d}.png")
    print(f"{len(prs.slides)} slides rendered to {PREV}")
    print("\n".join(problems) if problems else "no overflow found")


if __name__ == "__main__":
    sys.exit(main())
