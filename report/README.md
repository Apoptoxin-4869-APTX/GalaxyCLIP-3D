# Project report and supervisor deck

Builds `outputs/project_report/GalaxyCLIP3D_report.pdf` and `GalaxyCLIP3D_supervisor_deck.pptx` from the saved
results. Every number comes from one file, `outputs/project_report/numbers.json`, so the two documents cannot disagree.

| Step | Script | Output |
|---|---|---|
| collect the numbers | `collect_numbers.py` | `numbers.json` (thesis tables from `outputs/clip/report/report.md`, Phase D files, dataset index) |
| figures | `figures.py` | `figures/` (thesis figures at 300 dpi, pipeline, model, dataset, dark-matter controls) |
| report | `build_pdf.py` | `GalaxyCLIP3D_report.pdf` (ReportLab; equations via matplotlib mathtext) |
| deck | `build_deck.js` | `GalaxyCLIP3D_supervisor_deck.pptx` (pptxgenjs; theme, layouts, sections, speaker notes) |
| previews | `preview_deck.py` | `deck_preview/slide-NN.png` and a text-fit check (approximate; no LibreOffice needed) |
| consistency check | `check.py` | key numbers present in both documents |

```bash
cd GalaxyCLIP_3D
export PYTHONPATH=~/tools/report_pylibs:pylibs:.        # reportlab, python-pptx, lxml (pip --target), numpy/h5py
PY="/cluster/home/magilolg/venvs/tts-fish/bin/python -s"
$PY report/collect_numbers.py && $PY report/figures.py && $PY report/build_pdf.py
NODE_PATH=~/tools/report_node/node_modules node report/build_deck.js    # npm install pptxgenjs@3.12.0 there
$PY report/preview_deck.py && $PY report/check.py
```

The matched dark-matter control (c30) is picked up automatically once `outputs/clip/phase_d/dm_subset_compare.json`
exists; its written interpretation lives in `c30_reading.json`.

The title pages take the author and affiliation from `report/title.local.json`, which is not in git (without it they
are left out): `{"author": "...", "affiliation": "...", "affiliation_short": "..."}`. `build_deck.js` also uses the
PowerPoint skill's `apply_theme.js` (path in `SKILL` at the top).
