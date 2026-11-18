"""Check that the key numbers of numbers.json appear in the PDF report and in the deck (slide text, tables, notes and
chart data), so that neither document quotes a stale or mistyped value.

    cd GalaxyCLIP_3D && PYTHONPATH=~/tools/report_pylibs:pylibs:. python -s report/check.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pptx import Presentation
from pypdf import PdfReader

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/project_report"


def pdf_text() -> str:
    return " ".join(p.extract_text() or "" for p in PdfReader(str(OUT / "GalaxyCLIP3D_report.pdf")).pages)


def deck_text() -> tuple[str, list[float]]:
    prs = Presentation(str(OUT / "GalaxyCLIP3D_supervisor_deck.pptx"))
    parts, values = [], []
    for s in prs.slides:
        for sh in s.shapes:
            if sh.has_text_frame:
                parts.append(sh.text_frame.text)
            if getattr(sh, "has_table", False) and sh.has_table:
                parts += [c.text for r in sh.table.rows for c in r.cells]
            if getattr(sh, "has_chart", False) and sh.has_chart:
                for pl in sh.chart.plots:
                    for se in pl.series:
                        values += [v for v in se.values if v is not None]
        if s.has_notes_slide:
            parts.append(s.notes_slide.notes_text_frame.text)
    return " ".join(parts), values


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("−", "-").replace(" ", " ").replace("\xa0", " "))


def main() -> int:
    n = json.loads((OUT / "numbers.json").read_text())
    t, d, dm = n["test"], n["dataset"], n["dm_control"]
    r, p = dm["retrieval"], dm["probes"]
    checks = {
        "test R@1": f"{t['r1_final']:.1f}", "interval low": f"{t['interval'][0]:.1f}",
        "interval high": f"{t['interval'][1]:.1f}", "Phase B test": f"{t['r1_base']:.1f}",
        "gain": f"{t['final_minus_base']:.1f}", "gain low": f"{t['diff_interval'][0]:.1f}",
        "gain high": f"{t['diff_interval'][1]:.1f}", "statistics + CCA": "3.9", "control aligned": "75.4",
        "control independent": "6.0", "selected": f"{d['selected']:,}", "usable": f"{d['usable_r5']:,}",
        "train": f"{d['splits']['train']['galaxies']:,}", "val": str(d["splits"]["val"]["galaxies"]),
        "test": str(d["splits"]["test"]["galaxies"]),
        "stars-DM stellar": f"{r['C21-stellar']['pairs']['stars-dm']:.1f}",
        "stars-DM halo": f"{r['c29-Rvmax']['pairs']['stars-dm']:.1f}",
        "DM stellar mass stellar": f"{p['C21-stellar']['dm']['props']['log_mstar_total']:.2f}",
        "DM stellar mass halo": f"{p['c29-Rvmax']['dm']['props']['log_mstar_total']:.2f}",
        "DM R½ stellar": f"{p['C21-stellar']['dm']['props']['rhalf_star_kpc']:.2f}",
        "DM R½ halo": f"{p['c29-Rvmax']['dm']['props']['rhalf_star_kpc']:.2f}",
        "DM mean halo": f"{p['c29-Rvmax']['dm']['mean_r2']:.2f}",
        "stats mean halo": f"{p['stats-Rvmax']['dm']['mean_r2']:.2f}",
    }
    if n.get("c30"):
        rc, pc = n["c30"]["retrieval"]["c30-subset"], n["c30"]["probes"]["c30-subset"]
        checks.update({"c30 stars-DM": f"{rc['pairs']['stars-dm']:.1f}",
                       "c30 DM stellar mass": f"{pc['dm']['props']['log_mstar_total']:.2f}",
                       "c30 DM R½": f"{pc['dm']['props']['rhalf_star_kpc']:.2f}",
                       "c30 mean R@1": f"{rc['r1_no_gas_hi']:.1f}"})
    pdf, (deck, chart_values) = norm(pdf_text()), deck_text()
    deck = norm(deck)
    bad = 0
    for name, v in checks.items():
        in_pdf = v in pdf
        in_deck = v in deck or any(abs(x - float(v.replace(",", ""))) < 0.051 for x in chart_values
                                   if v.replace(",", "").replace(".", "").isdigit())
        flag = "ok" if in_pdf and in_deck else "MISSING"
        bad += flag != "ok"
        print(f"{flag:8s} {name:26s} {v:>8s}  pdf {'yes' if in_pdf else 'no ':3s}  deck {'yes' if in_deck else 'no'}")
    for word in ("TODO", "lorem", "xxx", "[insert", "NaN", "undefined", "None"):
        for label, txt in (("pdf", pdf), ("deck", deck)):
            if word in txt:
                print(f"LEFTOVER {word!r} in the {label}")
                bad += 1
    print("all key numbers found in both" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
