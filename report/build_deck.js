// Supervisor deck for GalaxyCLIP-3D, built with pptxgenjs from outputs/project_report/numbers.json and the report
// figures (report/figures.py). A structured deck: theme, slide layouts with placeholders, sections, speaker notes.
//
//   cd GalaxyCLIP_3D && NODE_PATH=~/tools/report_node/node_modules node report/build_deck.js
//   # -> outputs/project_report/GalaxyCLIP3D_supervisor_deck.pptx

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const SKILL = "/cluster/home/magilolg/.claude/remote/plugins/9d49db1a/066f54f6baceb2c5/skills/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const REPO = path.resolve(__dirname, "..");
const OUT = path.join(REPO, "outputs/project_report");
const FIG = path.join(OUT, "figures");
const DECK = path.join(OUT, "GalaxyCLIP3D_supervisor_deck.pptx");
const n = JSON.parse(fs.readFileSync(path.join(OUT, "numbers.json"), "utf8"));

const THEME = {
  name: "GalaxyCLIP",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0B1B33", lt1: "FFFFFF", dk2: "1F3A5F", lt2: "F3F2EE",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "EDA100", accent5: "6250D6", accent6: "E34948",
    hlink: "2A78D6", folHlink: "6250D6",
  },
};
// hex values for the hex-only options (chart colours, grid lines); the same as THEME and report/figures.py
const HEX = { navy: "1F3A5F", ink: "0B1B33", ink2: "52514E", muted: "898781", grid: "E1E0D9", tint: "F3F2EE",
  blue: "2A78D6", orange: "EB6834", aqua: "1BAF7A", yellow: "EDA100", violet: "6250D6", red: "E34948",
  green: "008300", gray: "B3B2AA", light: "D9D7CF" };
const MOD = { stars: ["stars", HEX.orange], dm: ["dark matter", HEX.violet], gas: ["gas", HEX.aqua], hi: ["HI", HEX.blue] };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "GalaxyCLIP-3D: project update";
const TITLE_FILE = path.join(__dirname, "title.local.json");   // author and affiliation; not in git (README.md)
const WHO = fs.existsSync(TITLE_FILE) ? JSON.parse(fs.readFileSync(TITLE_FILE, "utf8")) : {};
pres.author = WHO.author || "";
pres.subject = "Contrastive alignment of TNG50 galaxies as 3D point clouds";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------------------------------------------------------------- layouts
const W = 13.333, X0 = 0.6, CW = W - 2 * X0;
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.15, w: 11.7, h: 1.2, fontSize: 54, bold: true,
      color: C.background1, fontFace: THEME.headFontFace, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 2.45, w: 11.0, h: 1.0, fontSize: 22,
      color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.2, fontSize: 44, bold: true,
      color: C.background1, fontFace: THEME.headFontFace, valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.8, w: 11.7, h: 1.0, fontSize: 20,
      color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
for (const [name, tag] of [["CONTENT", ""], ["BACKUP", "backup · "]]) {
  pres.defineSlideMaster({
    title: name,
    background: { color: C.background1 },
    margin: [0.4, 0.6, 0.6, 0.6],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: X0, y: 0.32, w: CW, h: 0.95, fontSize: 34, bold: true,
        color: C.text2, fontFace: THEME.headFontFace, valign: "middle", margin: 0 }, text: "" } },
      { text: { text: `${tag}GalaxyCLIP-3D · project update`, options: { x: X0, y: 7.02, w: 6, h: 0.3, fontSize: 10,
        color: C.text2, margin: 0, transparency: 35 } } },
    ],
    slideNumber: { x: 12.15, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.text2, align: "right" },
  });
}

// ---------------------------------------------------------------- helpers
let objCount = 0;
const oname = (s) => `${s}-${++objCount}`;
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function image(slide, file, { x, y, w, h }) {
  // fit inside the box, keep the aspect ratio, centre
  const s = pngSize(file), ar = s.w / s.h;
  let iw = w, ih = w / ar;
  if (ih > h) { ih = h; iw = h * ar; }
  slide.addImage({ path: file, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih, objectName: oname("image"),
    altText: path.basename(file) });
  return { x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih };
}
function text(slide, t, o) {
  slide.addText(t, { isTextBox: true, margin: 0, fontSize: 16, color: C.text1, valign: "top", objectName: oname("text"),
    ...o });
}
function bullets(slide, items, o) {
  const runs = items.map((it, i) => {
    const parts = Array.isArray(it) ? it : [{ text: it }];
    return parts.map((p, j) => ({ text: p.text, options: { ...(p.options || {}),
      ...(j === 0 ? { bullet: { indent: 16 } } : {}), ...(j === parts.length - 1 && i < items.length - 1 ? { breakLine: true } : {}) } }));
  }).flat();
  slide.addText(runs, { isTextBox: true, margin: 0, fontSize: 16, color: C.text1, valign: "top", paraSpaceAfter: 7,
    objectName: oname("bullets"), ...o });
}
function card(slide, { x, y, w, h, title, body, accent, size = 14 }) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 },
    line: { color: C.background2 }, objectName: oname("card") });
  let ty = y + 0.18;
  if (accent) {
    slide.addShape(pres.shapes.OVAL, { x: x + 0.22, y: ty + 0.06, w: 0.2, h: 0.2, fill: { color: accent },
      line: { color: accent }, objectName: oname("dot") });
  }
  text(slide, title, { x: x + (accent ? 0.52 : 0.22), y: ty, w: w - (accent ? 0.74 : 0.44), h: 0.36, fontSize: 16,
    bold: true, color: C.text2 });
  if (Array.isArray(body)) {
    bullets(slide, body, { x: x + 0.22, y: ty + 0.48, w: w - 0.44, h: h - 0.75, fontSize: size, paraSpaceAfter: 4 });
  } else {
    text(slide, body, { x: x + 0.22, y: ty + 0.48, w: w - 0.44, h: h - 0.75, fontSize: size, color: C.text1 });
  }
}
function stat(slide, { x, y, w, value, label, color }) {
  text(slide, value, { x, y, w, h: 0.85, fontSize: 48, bold: true, color: color || C.accent1, fontFace: THEME.headFontFace });
  text(slide, label, { x, y: y + 0.88, w, h: 0.75, fontSize: 13, color: C.text1 });
}
const AXIS = { catAxisLabelColor: HEX.ink2, valAxisLabelColor: HEX.muted, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: HEX.grid, size: 0.5 },
  catGridLine: { style: "none" }, valAxisLineShow: false, catAxisLineShow: true, legendFontFace: "+mn-lt",
  legendFontSize: 12, legendColor: HEX.ink2, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: HEX.ink2,
  titleFontFace: "+mn-lt" };
function bar(slide, { x, y, w, h, labels, series, horizontal = false, max, min = 0, fmt = "0.0", legend, stacked = false,
  valueTitle, labelPos }) {
  const data = series.map((s) => ({ name: s.name, labels, values: s.values }));
  slide.addChart(pres.charts.BAR, data, { x, y, w, h, barDir: horizontal ? "bar" : "col",
    barGrouping: stacked ? "stacked" : "clustered", chartColors: series.map((s) => s.color), ...AXIS,
    showValue: true, dataLabelPosition: labelPos || (stacked ? "inEnd" : "outEnd"), dataLabelFormatCode: fmt,
    valAxisMinVal: min, valAxisMaxVal: max, valAxisLabelFormatCode: "0", barGapWidthPct: 55, showLegend: !!legend,
    legendPos: legend || "t", showValAxisTitle: !!valueTitle, valAxisTitle: valueTitle || "",
    valAxisTitleFontSize: 11, valAxisTitleColor: HEX.muted, valAxisTitleFontFace: "+mn-lt",
    catAxisOrientation: horizontal ? "maxMin" : "minMax", objectName: oname("chart") });
}
function table(slide, header, rows, { x, y, w, colW, fontSize = 12, boldLast = false, alignFrom = 1 }) {
  const head = header.map((t, j) => ({ text: t, options: { bold: true, color: C.text2, fontSize, align: j >= alignFrom ? "right" : "left",
    border: [{ type: "none" }, { type: "none" }, { pt: 1, color: HEX.ink }, { type: "none" }] } }));
  const body = rows.map((r, i) => r.map((t, j) => ({ text: t, options: { fontSize, color: C.text1,
    bold: boldLast && i === rows.length - 1, align: j >= alignFrom ? "right" : "left",
    border: [{ type: "none" }, { type: "none" }, { pt: 0.5, color: HEX.light }, { type: "none" }] } })));
  slide.addTable([head, ...body], { x, y, w, colW, margin: [0.03, 0.06, 0.03, 0.06], valign: "middle",
    objectName: oname("table") });
}
const f1 = (v) => v.toFixed(1), f2 = (v) => v.toFixed(2);
const T = n.tables, t = n.test, d = n.dataset, dm = n.dm_control, c30 = n.c30, fx = n.facts;
const row = (name, key) => T[name].rows.find((r) => r[0].replace(/&nbsp;/g, "").trim() === key);
const num = (s) => parseFloat(String(s).replace("−", "-").replace("+", ""));

function slide(master, sectionTitle, title) {
  const s = pres.addSlide({ masterName: master, sectionTitle });
  if (title) s.addText(title, { placeholder: "title" });
  return s;
}

// ---------------------------------------------------------------- 1 title
pres.addSection({ title: "Introduction" });
{
  const s = slide("TITLE", "Introduction", "GalaxyCLIP-3D");
  s.addText("Contrastive alignment of stars, dark matter, gas and HI in TNG50 galaxies as 3D point clouds",
    { placeholder: "body" });
  text(s, [WHO.author, WHO.affiliation_short, "project update, November 2026"].filter(Boolean).join(" · "),
    { x: 0.8, y: 3.55, w: 11.0, h: 0.4, fontSize: 16,
    color: C.background2 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.8, y: 4.25, w: 11.73, h: 2.85, rectRadius: 0.1,
    fill: { color: C.background1 }, line: { color: C.background1 }, objectName: oname("frame") });
  image(s, path.join(FIG, "example_galaxy.png"), { x: 0.95, y: 4.35, w: 11.43, h: 2.65 });
  s.addNotes("This is a status update on the 3D part of GalaxyCLIP. In one sentence: I built a point-cloud dataset of " +
    "about five thousand TNG50 galaxies, trained a CLIP-style model that maps a galaxy's stars, dark matter, gas and HI " +
    "into one embedding space, and tested carefully what that model has and has not learned. The picture shows one " +
    "galaxy as the model sees it: four clouds of 2,048 points each. Plan for the talk: data, method, results, two " +
    "controls on the dark matter, then limitations and the questions I would like your input on.");
}

// ---------------------------------------------------------------- 2 idea and result
{
  const s = slide("CONTENT", "Introduction", "One galaxy, four views, one embedding space");
  text(s, "Can a network tie together physically different components of the same galaxy?", { x: X0, y: 1.45, w: 6.6,
    h: 0.9, fontSize: 20, color: C.text2, italic: true });
  bullets(s, [
    [{ text: "Four views per galaxy: ", options: { bold: true } }, { text: "stars, dark matter, gas and HI, each a 3D cloud of 2,048 simulation particles" }],
    [{ text: "One encoder per view, ", options: { bold: true } }, { text: "trained CLIP-style so that a galaxy's four clouds land close together" }],
    [{ text: "Tested by retrieval: ", options: { bold: true } }, { text: "given one cloud, find the same galaxy's other cloud among all test galaxies" }],
    [{ text: "Tested by probes: ", options: { bold: true } }, { text: "which physical properties can be read off the embeddings" }],
  ], { x: X0, y: 2.45, w: 6.6, h: 3.6 });
  stat(s, { x: 7.9, y: 1.5, w: 4.8, value: `${f1(t.r1_final)}%`, label: `test R@1: the right galaxy's other cloud is the top match (chance 0.25%)` });
  stat(s, { x: 7.9, y: 3.25, w: 4.8, value: "10×", color: C.accent2, label: "better than hand-made statistics with CCA (3.9%)" });
  stat(s, { x: 7.9, y: 5.0, w: 4.8, value: `+${f1(t.final_minus_base)} pts`, color: C.accent3,
    label: `over the first PointNet (95% interval ${f1(t.diff_interval[0])}–${f1(t.diff_interval[1])})` });
  s.addNotes(`The question is whether one representation can link components that are physically very different. ` +
    `Retrieval is the main test: I take a galaxy's stars, for example, and ask the model to find the same galaxy's dark ` +
    `matter among all 499 test galaxies. The final model gets the top match right ${f1(t.r1_final)}% of the time, averaged ` +
    `over ten search directions; chance is a quarter of a percent. Hand-made statistics aligned with CCA reach 3.9%, so ` +
    `the network does about ten times better, and it gained ${f1(t.final_minus_base)} points over my first PointNet during ` +
    `the ablations. The interval is a bootstrap over test galaxies.`);
}

// ---------------------------------------------------------------- 3 point clouds
pres.addSection({ title: "Data and method" });
{
  const s = slide("CONTENT", "Data and method", "From simulation particles to point clouds");
  image(s, path.join(FIG, "example_galaxy.png"), { x: X0, y: 1.35, w: CW, h: 3.75 });
  const feats = { stars: "position, velocity, age, metallicity", dm: "position, velocity", gas: "position, velocity, density, temperature, metallicity, B field, neutral fraction", hi: "position, velocity, neutral fraction" };
  const cw = (CW - 3 * 0.25) / 4;
  Object.entries(feats).forEach(([m, f], i) => card(s, { x: X0 + i * (cw + 0.25), y: 5.2, w: cw, h: 1.65,
    title: MOD[m][0], body: f, accent: MOD[m][1], size: 13 }));
  s.addNotes("Each galaxy is a Subfind subhalo of TNG50-1 at redshift zero. For each component the loader draws 2,048 " +
    "points from the galaxy's own particles inside five stellar half-mass radii. Every point carries its position and " +
    "velocity, plus the physics of that particle type: age and metallicity for stars, density, temperature, metallicity, " +
    "magnetic field and neutral fraction for gas. HI is drawn from the same gas cells but weighted by neutral hydrogen " +
    "mass. Positions are scaled by the sphere radius and velocities by the galaxy's maximum circular velocity, so the " +
    "model does not get the absolute size directly. Remember this scaling, it comes back in the dark-matter control.");
}

// ---------------------------------------------------------------- 4 dataset
{
  const s = slide("CONTENT", "Data and method", "A new dataset built from the raw simulation");
  const steps = [
    ["TNG50-1", `snapshot 99 (z = 0)\n${fx.raw_snapshot.size_tb} TB, fresh complete copy`],
    ["Selection", `${d.selected.toLocaleString("en")} galaxies\nin ${d.halos.toLocaleString("en")} halos`],
    ["Extraction", `particles within 10 R½\n${fx.storage.particles_billion} billion, ${fx.storage.shards_gb} GB`],
    ["Splits by halo", `80 / 10 / 10, consistent\nacross snapshots`],
    ["Loader", "4 clouds × 2,048 points\nnew draw every epoch"],
  ];
  const bw = 2.15, gap = (CW - 5 * bw) / 4;
  steps.forEach(([h, b], i) => {
    const x = X0 + i * (bw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.55, w: bw, h: 1.55, rectRadius: 0.08, fill: { color: C.background2 },
      line: { color: C.background2 }, objectName: oname("step") });
    text(s, h, { x: x + 0.12, y: 1.7, w: bw - 0.24, h: 0.4, fontSize: 16, bold: true, color: C.text2, align: "center" });
    text(s, b, { x: x + 0.12, y: 2.15, w: bw - 0.24, h: 0.85, fontSize: 13, align: "center" });
    if (i < steps.length - 1) {
      s.addShape(pres.shapes.RIGHT_TRIANGLE, { x: x + bw + gap / 2 - 0.09, y: 2.24, w: 0.18, h: 0.18, rotate: 45,
        fill: { color: HEX.muted }, line: { color: HEX.muted }, objectName: oname("arrow") });
    }
  });
  const sp = d.splits;
  const sw = (CW - 2 * 0.4) / 3;
  stat(s, { x: X0, y: 3.55, w: sw, value: d.selected.toLocaleString("en"),
    label: `galaxies selected: ${d.selected_centrals.toLocaleString("en")} centrals, ${d.selected_satellites.toLocaleString("en")} satellites` });
  stat(s, { x: X0 + sw + 0.4, y: 3.55, w: sw, value: d.usable_r5.toLocaleString("en"), color: C.accent3,
    label: `usable at 5 R½: ${sp.train.galaxies.toLocaleString("en")} train, ${sp.val.galaxies} validation, ${sp.test.galaxies} test (whole halos)` });
  stat(s, { x: X0 + 2 * (sw + 0.4), y: 3.55, w: sw, value: `${Math.round(100 * d.gas_off_r5 / d.usable_r5)}%`, color: C.accent2,
    label: "of usable galaxies have too little gas for a gas cloud; it is masked" });
  text(s, `An older copy of the snapshot on the cluster lacked ${fx.raw_snapshot.old_copy_missing_files} files, so snapshot 99 was downloaded again (${fx.raw_snapshot.download}). ` +
    `Large galaxies are thinned to about 262,000 particles per type, with weights that keep later draws unbiased.`,
    { x: X0, y: 5.75, w: CW, h: 0.9, fontSize: 13, color: C.text2 });
  s.addNotes(`The existing copy of snapshot 99 on the cluster was incomplete, so I downloaded a fresh one, 2.9 terabytes, ` +
    `and wrote the whole pipeline: selection, extraction into shards, splits, frozen evaluation clouds and quality ` +
    `checks. The selection matches the 2D maps: real galaxies with at least 4,096 star and dark-matter particles and a ` +
    `resolved size. That gives ${d.selected.toLocaleString("en")} galaxies, ${d.usable_r5.toLocaleString("en")} usable at the default sphere. Splits ` +
    `are by halo, and for future snapshots by the halo a galaxy ends up in at z = 0, so a galaxy and its progenitor ` +
    `can never sit in different splits. About a third of the galaxies, mostly small satellites, have too little gas; ` +
    `they stay in with the gas and HI clouds masked.`);
}

// ---------------------------------------------------------------- 5 model
{
  const s = slide("CONTENT", "Data and method", "One encoder per modality, one shared space");
  image(s, path.join(FIG, "model.png"), { x: X0 - 0.1, y: 1.4, w: 8.2, h: 5.4 });
  const c = n.final_config;
  card(s, { x: 8.75, y: 1.5, w: 3.98, h: 5.25, title: "Final setup", size: 14, body: [
    `PointNet per modality, widths ${c.pointnet_widths.join("–")}, ${fx.compute.params_m} M parameters`,
    "contrastive loss (InfoNCE) over all 6 pairs; temperature learned",
    "own random rotation and mirror for every cloud",
    `batch ${c.batch_size}, ${c.epochs} epochs, AdamW, cosine schedule`,
    "one H100 GPU, about 2–3 hours per run",
  ] });
  s.addNotes("Each modality has its own PointNet: the same small network on every point, a maximum over points, and a " +
    "projection to a 512-dimensional unit vector. The loss is the symmetric InfoNCE of CLIP, computed for all six pairs " +
    "of modalities and averaged; pairs are computed only on galaxies that have both clouds, so galaxies without gas still " +
    "train the stars and dark matter pair. The temperature is learned. Every cloud gets its own random rotation; I will " +
    "show in two slides why that matters. The final setup came out of the ablations: wider encoders, learned temperature " +
    "and 600 epochs.");
}

// ---------------------------------------------------------------- 6 scoring
{
  const s = slide("CONTENT", "Data and method", "How the model is scored");
  const cw = (CW - 2 * 0.3) / 3;
  card(s, { x: X0, y: 1.5, w: cw, h: 4.9, title: "Retrieval, R@1", accent: HEX.blue, body: [
    "query: one galaxy in one modality",
    "gallery: all test galaxies in another modality (499 for dark matter)",
    "hit: the true galaxy is the single best match; ties count against",
    "headline: mean of 10 directions; gas–HI left out (same gas cells)",
  ] });
  card(s, { x: X0 + cw + 0.3, y: 1.5, w: cw, h: 4.9, title: "Linear probes", accent: HEX.aqua, body: [
    "ridge regression on frozen embeddings",
    "18 properties (R²): masses, star formation, metallicity, colour, age, size, spin …",
    "2 yes/no properties (AUC): central, quenched",
    "references: hand-made statistics of the same cloud; the untrained network",
  ] });
  card(s, { x: X0 + 2 * (cw + 0.3), y: 1.5, w: cw, h: 4.9, title: "Protocol", accent: HEX.violet, body: [
    "every choice made on the 520 validation galaxies",
    "checkpoint with the best validation R@1",
    "test set (499 galaxies) scored once, on 16 November, after the setup was fixed",
    "95% intervals from bootstrapping the test galaxies",
  ] });
  s.addNotes("Two kinds of evaluation. Retrieval: each galaxy is a query in one modality and searched among all galaxies " +
    "in another, not just inside a batch. R@1 counts the top match only, and ties count against, so a collapsed model " +
    "scores zero. Gas and HI are drawn from the same cells, so that pair is almost trivial; the headline number leaves it " +
    "out. Probes: linear regressions on frozen embeddings for 18 properties, compared with hand-made statistics of the " +
    "same cloud and with the untrained network. All choices were made on validation; the test set was scored exactly " +
    "once.");
}

// ---------------------------------------------------------------- 7 baselines
pres.addSection({ title: "Results" });
{
  const s = slide("CONTENT", "Results", "Without learning, the floor is low");
  const b = n.baselines_test;
  bar(s, { x: X0, y: 1.45, w: 7.6, h: 5.2, horizontal: true, max: 25, labels: ["spin direction, shared frame",
    "spin direction, independent rotations", "statistics + CCA", "chance"], series: [{ name: "test R@1 (%)",
    values: [b["orientation: spin direction, shared frame"].r1_10dir, b["orientation: spin direction, independent rotations"].r1_10dir,
      b["statistics + CCA"].r1_10dir, 0.25], color: HEX.navy }], fmt: "0.0", valueTitle: "test R@1 (%), 10 directions" });
  bullets(s, [
    [{ text: "Orientation alone is a strong cue: ", options: { bold: true } }, { text: "matching only the spin direction finds the right galaxy for 1 in 5 queries when the clouds share a frame" }],
    [{ text: "Independent rotations remove it: ", options: { bold: true } }, { text: "the same rule drops to chance" }],
    [{ text: "About 20 statistics per cloud + CCA: ", options: { bold: true } }, { text: "3.9%, the reference the network has to beat" }],
  ], { x: 8.55, y: 1.7, w: 4.18, h: 4.8, fontSize: 15 });
  s.addNotes("Before training anything I measured two baselines. If the clouds keep the simulation's orientation, simply " +
    "comparing the direction of the stars' and the gas's angular momentum already finds the right galaxy almost a fifth of " +
    "the time, because a galaxy's components rotate together. With an independent rotation per cloud that drops to " +
    "chance. The honest baseline is the second: about twenty orientation-free statistics per cloud, aligned with " +
    "canonical correlation analysis, gives 3.9% on test.");
}

// ---------------------------------------------------------------- 8 orientation shortcut
{
  const s = slide("CONTENT", "Results", "A shortcut we had to close");
  const ctl = row("orientation_shortcut", "one rotation shared by a galaxy's clouds (control)");
  const pb = row("orientation_shortcut", "independently rotated clouds (Phase B PointNet)");
  bar(s, { x: X0, y: 1.45, w: 7.4, h: 5.2, labels: ["clouds share one rotation", "clouds rotated independently"],
    series: [{ name: "trained on shared rotations (control)", values: [num(ctl[1]), num(ctl[2])], color: HEX.aqua },
      { name: "trained on independent rotations", values: [num(pb[1]), num(pb[2])], color: HEX.blue }],
    max: 80, legend: "t", valueTitle: "validation R@1 (%), 10 directions" });
  card(s, { x: 8.4, y: 1.6, w: 4.33, h: 2.4, title: "What happened", accent: HEX.aqua, size: 14,
    body: `Trained on clouds that share an orientation, a PointNet scores ${ctl[1]}% (test ${ctl[3]}%), but ${ctl[2]}% once the clouds are rotated independently: it compared orientations, not galaxies.` });
  card(s, { x: 8.4, y: 4.2, w: 4.33, h: 2.45, title: "Consequence", accent: HEX.blue, size: 14,
    body: "Every cloud gets its own random rotation. Any new modality, including the 2D maps, needs the same check." });
  s.addNotes(`This was the most important methodological finding. The first version of the loader gave all four clouds of a ` +
    `galaxy the same random rotation. A model trained like that reached over 70% on validation, which looked great, but ` +
    `scored on independently rotated clouds it fell to ${ctl[2]}%: it had learned to match orientations. With independent ` +
    `rotations the model scores the same either way, so it cannot use that cue. All results from here on use independent ` +
    `rotations, and I think the 2D model should get the same test.`);
}

// ---------------------------------------------------------------- 9 first models
{
  const s = slide("CONTENT", "Results", "PointNet beats a patch transformer");
  const pn = n.curves.pointnet_b.filter((p) => p[0] <= 300), pt = n.curves.patch_b.filter((p) => p[0] <= 300);
  s.addChart(pres.charts.LINE, [
    { name: "PointNet (1.2 M parameters per modality)", labels: pn.map((p) => String(p[0])), values: pn.map((p) => p[1]) },
    { name: "patch transformer (11.9 M per modality)", labels: pn.map((p) => String(p[0])), values: pt.map((p) => p[1]) },
  ], { x: X0, y: 1.45, w: 8.1, h: 5.25, ...AXIS, chartColors: [HEX.orange, HEX.violet], lineSize: 2, lineDataSymbol: "none",
    showLegend: true, legendPos: "t", valAxisMinVal: 0, valAxisMaxVal: 40, catAxisLabelFrequency: 10,
    showCatAxisTitle: true, catAxisTitle: "epoch", catAxisTitleFontSize: 11, catAxisTitleColor: HEX.muted,
    showValAxisTitle: true, valAxisTitle: "validation R@1 (%), 10 directions", valAxisTitleFontSize: 11,
    valAxisTitleColor: HEX.muted, objectName: oname("chart") });
  const pb = row("phase_c_ablations", "Phase B PointNet"), ptr = row("phase_c_ablations", "Phase B patch transformer (reference)");
  bullets(s, [
    [{ text: `PointNet: ${pb[1]}% `, options: { bold: true } }, { text: `(best epoch ${pb[4]}), still rising slowly` }],
    [{ text: `Patch transformer: peak ${ptr[1]}% `, options: { bold: true } }, { text: `at epoch ${ptr[4]}, then falls to ${ptr[5]}%: it memorises the ~4,000 training galaxies` }],
    [{ text: "Smaller, regularised transformers ", options: { bold: true } }, { text: "reached 27.8% at best, so PointNet was kept" }],
  ], { x: 9.0, y: 1.7, w: 3.73, h: 4.9, fontSize: 15 });
  s.addNotes("Phase B compared two encoders. PointNet is small and treats points independently before pooling; the patch " +
    "transformer, in the style of Point-MAE, groups points into local patches and runs attention over them. With only " +
    "about four thousand training galaxies the transformer overfits: it peaks early and then its validation score " +
    "collapses while the training loss keeps falling. Smaller and regularised versions helped but never caught up, so " +
    "the rest of the project uses PointNet.");
}

// ---------------------------------------------------------------- 10 ablations
{
  const s = slide("CONTENT", "Results", "Ablations: three small gains add up");
  const pick = [["learned temperature + wide + 600 epochs (3 seeds)", "combined (3 seeds)"],
    ["+ rotation-free point features", "rotation-free point features"], ["learned temperature", "learned temperature"],
    ["wide PointNet (9.1 M; base 4.9 M)", "wide encoder"], ["600 epochs (instead of 300)", "600 epochs"],
    ["4,096 points per cloud", "4,096 points"], ["weight averaging, EMA 0.99", "EMA 0.99"],
    ["1,024 points per cloud (instead of 2,048)", "1,024 points"], ["narrow PointNet (3.3 M parameters)", "narrow encoder"],
    ["batch 512", "batch 512"], ["one encoder shared by all modalities", "one shared encoder"], ["no velocities", "no velocities"]];
  const vals = pick.map(([k]) => num(row("phase_c_ablations", k)[2]));
  bar(s, { x: X0, y: 1.4, w: 8.3, h: 5.35, horizontal: true, stacked: true, min: -12, max: 6, labels: pick.map((p) => p[1]),
    series: [{ name: "better", values: vals.map((v) => (v > 0 ? v : 0)), color: HEX.blue },
      { name: "worse", values: vals.map((v) => (v < 0 ? v : 0)), color: HEX.gray }],
    fmt: '+0.0;-0.0;;', labelPos: "inEnd", valueTitle: "change in best validation R@1 (points) vs Phase B PointNet (33.2%)" });
  bullets(s, [
    [{ text: "Rule set before the runs: ", options: { bold: true } }, { text: "a change counts above ±2 points (two seeds differ by 0.3)" }],
    [{ text: "Final setup: ", options: { bold: true } }, { text: "learned temperature, wide encoder and 600 epochs, each below the threshold alone, together +4.8" }],
    [{ text: "Velocities matter most: ", options: { bold: true } }, { text: "without them stars–dark matter almost vanishes" }],
  ], { x: 9.15, y: 1.7, w: 3.58, h: 4.9, fontSize: 14.5 });
  s.addNotes(`Phase C changed one thing at a time, ${fx.phase_c_runs} runs in total. I fixed the decision rule in advance: a change ` +
    `counts if it moves the best validation score by more than two points. Three changes were just below that line, a ` +
    `learned temperature, a wider encoder and longer training, and they add up to almost five points over three seeds. ` +
    `Rotation-free point features were the best single change but redundant once combined. Larger batches, a narrower ` +
    `or shared encoder and fewer points hurt. Removing velocities costs almost twelve points and kills the dark-matter ` +
    `pairs, which tells us the kinematics carry most of the cross-modal signal.`);
}

// ---------------------------------------------------------------- 11 test result
{
  const s = slide("CONTENT", "Results", `Final model: ${f1(t.r1_final)}% R@1 on the test set`);
  stat(s, { x: X0, y: 1.5, w: 3.6, value: `${f1(t.r1_final)}%`, label: `test R@1, 3 seeds (± 0.2); 95% interval ${f1(t.interval[0])}–${f1(t.interval[1])}` });
  stat(s, { x: X0, y: 3.2, w: 3.6, value: "74.9%", color: C.accent3, label: "test R@10: the right galaxy among the top 10 of ~500" });
  stat(s, { x: X0, y: 4.9, w: 3.6, value: `+${f1(t.final_minus_base)}`, color: C.accent2, label: `points over the first PointNet (${f1(t.r1_base)}%)` });
  const rp = T.retrieval_pairs, labels = ["stars–DM", "stars–gas", "stars–HI", "DM–gas", "DM–HI", "gas–HI"];
  const get = (k) => rp.rows.find((r) => r[0] === k).slice(1).map(num);
  bar(s, { x: 4.45, y: 1.4, w: 8.28, h: 5.35, labels, max: 100, legend: "t", valueTitle: "test R@1 (%), mean of both directions",
    series: [{ name: "final setup", values: get("final setup (3 seeds)"), color: HEX.blue },
      { name: "first PointNet", values: get("Phase B PointNet"), color: HEX.orange },
      { name: "statistics + CCA", values: get("statistics + CCA"), color: HEX.gray }] });
  s.addNotes(`On the 499 test galaxies the final setup reaches ${f1(t.r1_final)}% over the ten directions, plus or minus 0.2 ` +
    `over three seeds, and the right galaxy is in the top ten three quarters of the time. Per pair, stars and gas match ` +
    `best; the dark matter is the hardest modality, but the gains are largest there: stars to dark matter almost doubled ` +
    `compared with the first model, from 12.8 to 24.1%. Gas to HI is near-duplicate and only shown for completeness. The ` +
    `test score is about two points above validation, which is within the sampling uncertainty of five hundred galaxies.`);
}

// ---------------------------------------------------------------- 12 probes
{
  const s = slide("CONTENT", "Results", "The embeddings carry the galaxy's physics");
  const ps = T.probes_summary.rows, g = (k) => ps.find((r) => r[0] === k).slice(1).map((v) => num(v.split(" ")[0]));
  bar(s, { x: X0, y: 1.4, w: 7.9, h: 5.35, labels: ["stars", "dark matter", "gas", "HI"], max: 1, fmt: "0.00", legend: "t",
    valueTitle: "mean test R², 18 properties",
    series: [{ name: "final setup", values: g("final setup (3 seeds)"), color: HEX.blue },
      { name: "hand-made statistics", values: g("hand-made statistics"), color: HEX.gray },
      { name: "untrained network", values: g("untrained network"), color: HEX.light }] });
  card(s, { x: 8.85, y: 1.55, w: 3.88, h: 2.55, title: "Dark matter gains most", accent: HEX.violet, size: 14,
    body: "From the dark-matter cloud alone: stellar mass R² 0.80 (statistics 0.29), black-hole mass 0.79 (0.39), stellar metallicity 0.78 (0.30)." });
  card(s, { x: 8.85, y: 4.3, w: 3.88, h: 2.4, title: "Why", accent: HEX.blue, size: 14,
    body: "The other modalities supervise each encoder, so it learns what matters about the galaxy, not just about its own cloud." });
  s.addNotes("Linear probes on the frozen embeddings predict eighteen galaxy properties much better than hand-made statistics " +
    "of the same clouds. For stars and gas the statistics are already good, so the gain is moderate. For HI and especially " +
    "the dark matter the gain is large: from the dark-matter cloud alone the embedding predicts stellar mass with R squared " +
    "0.80, against 0.29 for statistics. That result looked almost too good, which led to the control on the next slide.");
}

// ---------------------------------------------------------------- 13 DM frame control
pres.addSection({ title: "Dark-matter controls" });
{
  const s = slide("CONTENT", "Dark-matter controls", "Is the dark-matter result real? Frame control");
  text(s, [{ text: "Every cloud, dark matter included, is cut and scaled by 5 stellar half-mass radii, so the dark-matter cloud carries " +
    "the stars' size. Control (c29): cut and scale the dark matter by the halo's own radius R" }, { text: "vmax", options: { subscript: true } },
  { text: " instead." }], { x: X0, y: 1.35, w: CW, h: 0.75, fontSize: 15, color: C.text2 });
  const r = dm.retrieval, pr = dm.probes;
  const pairs = [["stars-dm", "stars–DM"], ["dm-gas", "DM–gas"], ["dm-hi", "DM–HI"], ["stars-gas", "stars–gas"], ["stars-hi", "stars–HI"]];
  bar(s, { x: X0, y: 2.2, w: 5.9, h: 4.55, labels: pairs.map((p) => p[1]), max: 100, legend: "t",
    valueTitle: `R@1 (%), same ${dm.shared.val} validation galaxies`,
    series: [{ name: "stellar frame (C21)", values: pairs.map(([k]) => r["C21-stellar"].pairs[k]), color: HEX.blue },
      { name: "halo frame (c29)", values: pairs.map(([k]) => r["c29-Rvmax"].pairs[k]), color: HEX.red }] });
  const props = [["log_mstar_total", "stellar mass"], ["rhalf_star_kpc", "half-mass radius"], ["gas_fraction", "gas fraction"],
    ["log_mdm_total", "DM mass"], ["log_mbh", "black-hole mass"], ["log_z_stars", "metallicity"]];
  bar(s, { x: 6.75, y: 2.2, w: 5.98, h: 4.55, labels: props.map((p) => p[1]), max: 1, fmt: "0.00", legend: "t",
    valueTitle: "dark-matter embedding, validation R²",
    series: [{ name: "stellar frame (C21)", values: props.map(([k]) => pr["C21-stellar"].dm.props[k]), color: HEX.blue },
      { name: "halo frame (c29)", values: props.map(([k]) => pr["c29-Rvmax"].dm.props[k]), color: HEX.red },
      { name: "statistics, halo frame", values: props.map(([k]) => pr["stats-Rvmax"].dm.props[k]), color: HEX.gray }] });
  s.addNotes(`Because every cloud is scaled by the stars' half-mass radius, the shape of the dark-matter cloud tells the model ` +
    `how big the stellar body is, and size correlates with stellar mass. To test this I retrained the final setup with ` +
    `the dark matter cut at R_vmax, the radius where the halo's rotation curve peaks. On the same ${dm.shared.val} validation ` +
    `galaxies, stars to dark matter retrieval halves, from ${f1(r["C21-stellar"].pairs["stars-dm"])} to ${f1(r["c29-Rvmax"].pairs["stars-dm"])}%, while pairs ` +
    `without dark matter are unchanged. The clearest sign of the leak is the half-mass radius: predicted with ` +
    `${f2(pr["C21-stellar"].dm.props.rhalf_star_kpc)} before, ${f2(pr["c29-Rvmax"].dm.props.rhalf_star_kpc)} now. Stellar mass drops from ${f2(pr["C21-stellar"].dm.props.log_mstar_total)} to ` +
    `${f2(pr["c29-Rvmax"].dm.props.log_mstar_total)}, but dark-matter mass, black-hole mass and metallicity hold, and the trained embedding still ` +
    `beats statistics of the same clouds, ${f2(pr["c29-Rvmax"].dm.mean_r2)} against ${f2(pr["stats-Rvmax"].dm.mean_r2)} mean R squared. Caveats: one seed, and the ` +
    `control trained on 80% of the galaxies, which the next slide addresses.`);
}

// ---------------------------------------------------------------- 14 matched control
{
  const s = slide("CONTENT", "Dark-matter controls", c30 ? "Matched control: the frame caused the drop" :
    "Matched control: same galaxies, stellar frame");
  const sub = n.rvmax_subset;
  if (!c30) {
    card(s, { x: X0, y: 1.55, w: 6.0, h: 5.1, title: "Why this run", accent: HEX.green, size: 15, body: [
      `the halo-frame run differs from the final setup in two ways: the frame and the training set (${sub.splits.train.toLocaleString("en")} of 4,052 galaxies)`,
      "c30 trains the final setup in the usual stellar frame on exactly those galaxies",
      "its validation clouds are the frozen clouds the final setup was scored on",
      "so it separates the effect of the frame from the effect of the smaller training set",
    ] });
    card(s, { x: 6.9, y: 1.55, w: 5.83, h: 2.4, title: "Status", accent: HEX.muted, size: 15,
      body: "Training job 168500 was waiting for an H100 when this deck was built; a scoring job runs automatically after it." });
    card(s, { x: 6.9, y: 4.15, w: 5.83, h: 2.5, title: "How to read it", accent: HEX.blue, size: 15, body: [
      "c30 ≈ C21 on the dark-matter numbers: the frame explains the drop",
      "c30 ≈ c29: the smaller training set explains it",
    ] });
  } else {
    // order c29 | C21 | c30 as in the report figure: red and green never side by side (colour-blind check)
    const r = c30.retrieval, pr = c30.probes, keys = ["c29-Rvmax", "C21-stellar", "c30-subset"].filter((k) => r[k]);
    const nm = { "C21-stellar": "C21: stellar frame, all galaxies", "c30-subset": "c30: stellar frame, same galaxies", "c29-Rvmax": "c29: halo frame" };
    const col = { "C21-stellar": HEX.blue, "c30-subset": HEX.green, "c29-Rvmax": HEX.red };
    bar(s, { x: X0, y: 1.45, w: 7.6, h: 5.25, labels: ["stars–DM R@1 (%)", "DM–gas R@1 (%)", "mean R@1 (%)"], max: 50, legend: "t",
      series: keys.map((k) => ({ name: nm[k], values: [r[k].pairs["stars-dm"], r[k].pairs["dm-gas"], r[k].r1_no_gas_hi], color: col[k] })) });
    table(s, ["DM embedding, R²", ...keys.map((k) => k.split("-")[0])], [
      ["stellar mass", ...keys.map((k) => f2(pr[k].dm.props.log_mstar_total))],
      ["half-mass radius", ...keys.map((k) => f2(pr[k].dm.props.rhalf_star_kpc))],
      ["mean of 18", ...keys.map((k) => f2(pr[k].dm.mean_r2))],
    ], { x: 8.5, y: 1.6, w: 4.23, colW: [1.83, 0.8, 0.8, 0.8], fontSize: 13, boldLast: true });
    card(s, { x: 8.5, y: 3.3, w: 4.23, h: 3.4, title: "Verdict", accent: HEX.green, size: 14, body: n.c30_reading_short || [] });
  }
  s.addNotes(c30 ? (n.c30_notes || "") : "The halo-frame run changed two things at once: the dark-matter frame and, because the " +
    "R_vmax sphere has to fit in the stored data, the training set, which is 80% of the galaxies. This matched control " +
    "trains the final setup in the normal stellar frame on exactly those galaxies and scores it on the same frozen " +
    "validation clouds. If it matches the final setup, the frame explains the drop; if it falls to the halo-frame numbers, " +
    "the smaller training set does.");
}

// ---------------------------------------------------------------- 15 limitations
pres.addSection({ title: "Wrap-up" });
{
  const s = slide("CONTENT", "Wrap-up", "Limitations");
  const items = [
    ["One snapshot", "all galaxies at z = 0; each further snapshot needs 2.93 TB of raw data before reduction", HEX.blue],
    ["Small dataset", "about 5,000 galaxies: the transformer overfits, larger models need more data", HEX.orange],
    ["2,048 points", "a small sample of galaxies with up to millions of particles; 4,096 points helped slightly", HEX.aqua],
    ["Bound particles only", "gas around a galaxy that is not bound to it is not in the clouds", HEX.yellow],
    ["Scaling leaks size", "the stellar radius imprints the stars' size on the dark-matter cloud" +
      (c30 ? "; two controls confirm it" : ""), HEX.violet],
    ["Test set now used", "controls ran with one seed; further choices must be made on validation", HEX.red],
  ];
  const cw = (CW - 2 * 0.3) / 3, ch = 2.2;
  items.forEach(([h, b, c], i) => card(s, { x: X0 + (i % 3) * (cw + 0.3), y: 1.5 + Math.floor(i / 3) * (ch + 0.3), w: cw,
    h: ch, title: h, body: b, accent: c, size: 15 }));
  s.addNotes("The main limitations. Everything is at redshift zero; the dataset can hold more snapshots, but storage is the " +
    "bottleneck. Five thousand galaxies is small for representation learning, which is why the transformer overfits. " +
    "2,048 points is a small sample of large galaxies. Only bound particles are used. The scaling issue I just showed. And " +
    "the test set has been used once, so from now on everything has to be decided on validation; a later snapshot is not " +
    "a fresh test set, because it contains the same galaxies at another time.");
}

// ---------------------------------------------------------------- 16 next steps and questions
{
  const s = slide("CONTENT", "Wrap-up", "Next steps and questions for you");
  card(s, { x: X0, y: 1.5, w: 6.4, h: 5.2, title: "Next steps", accent: HEX.blue, size: 15, body: [
    "2D↔3D alignment: freeze the 2D map model, export its embeddings, train the 3D encoders to match them (needs one common split)",
    c30 ? n.c30_next_step_short : "settle the dark-matter frame with the matched control",
    "more seeds for the controls; test the 2D model for the orientation shortcut",
    "add snapshot 91 when storage allows",
    "write the 3D thesis chapter (figures and tables are generated from saved results)",
  ] });
  card(s, { x: 7.3, y: 1.5, w: 5.43, h: 5.2, title: "Questions for you", accent: HEX.orange, size: 15, body: [
    "Which dark-matter frame should the thesis report: stellar frame with the control beside it, or the halo frame?",
    "What comes first: 2D↔3D alignment, more snapshots, or more seeds?",
    "Is the test protocol (scored once) enough, or should a new held-out set be planned?",
  ] });
  s.addNotes(c30 ? n.c30_next_step_notes : "For the next phase I see five items, with the 2D to 3D alignment as the main research step: keep the 2D model " +
    "frozen and train the 3D encoders to match its embeddings. Before that I need to settle the dark-matter frame. I would " +
    "like your view on three questions: which frame to report in the thesis, what to prioritise, and whether the " +
    "test protocol is sufficient as it is.");
}

// ---------------------------------------------------------------- backup
pres.addSection({ title: "Backup" });
{
  const s = slide("SECTION", "Backup", "Backup");
  s.addText("Details, full tables and code", { placeholder: "body" });
  s.addNotes("Backup slides with the full numbers, for questions.");
}
{
  const s = slide("BACKUP", "Backup", "Dataset construction");
  table(s, ["Step", "Detail"], [
    ["selection", "SubhaloFlag = 1; ≥ 4,096 bound star and DM particles; stellar half-mass radius ≥ 0.576 kpc (two softening lengths)"],
    ["galaxies", `${d.selected.toLocaleString("en")} (${d.selected_centrals.toLocaleString("en")} centrals, ${d.selected_satellites.toLocaleString("en")} satellites) in ${d.halos.toLocaleString("en")} halos; ${d.usable_r5.toLocaleString("en")} usable at 5 R½`],
    ["stored particles", `bound particles within 10 R½; at most about 262,000 per type, thinned with unbiased weights (${fx.storage.capped_galaxies} galaxies capped)`],
    ["size on disk", `${fx.storage.particles_billion} billion particles (stars ${fx.storage.stars_m} M, DM ${fx.storage.dm_m} M, gas ${fx.storage.gas_m} M), ${fx.storage.shards_gb} GB`],
    ["masks", `gas cloud needs ≥ 4,096 gas cells, HI ≥ 512 effective cells; ${d.gas_off_r5.toLocaleString("en")} galaxies without gas cloud, ${d.hi_off_r5} without HI`],
    ["splits", "whole halos, 80/10/10, balanced over stellar mass; later snapshots follow the z = 0 halo (SubLink)"],
    ["evaluation clouds", "frozen: one fixed draw and orientation per galaxy and modality"],
  ], { x: X0, y: 1.4, w: CW, colW: [2.0, CW - 2.0], fontSize: 12, alignFrom: 9 });
  image(s, path.join(FIG, "dataset.png"), { x: X0, y: 4.45, w: CW, h: 2.45 });
  s.addNotes("Details of the dataset build, for reference.");
}
{
  const s = slide("BACKUP", "Backup", "Hyperparameters and compute");
  const c = n.final_config;
  table(s, ["Setting", "Value"], [
    ["encoder", `PointNet per modality, widths ${c.pointnet_widths.join("–")}, ${fx.compute.params_m} M parameters in total`],
    ["embedding", `${c.embed_dim}-dimensional, unit length`],
    ["loss", "symmetric InfoNCE, mean over the 6 pairs, masked pairs skipped"],
    ["temperature", `learned, from ${c.temperature} to ${n.final_train.final_temperature.toFixed(3)}`],
    ["optimiser", `AdamW, learning rate ${c.lr}, weight decay ${c.weight_decay}`],
    ["schedule", `${c.warmup_epochs} warm-up epochs, cosine to 0, ${c.epochs} epochs, batch ${c.batch_size} (mass-balanced)`],
    ["precision", "bfloat16; ZClip adaptive gradient clipping"],
    ["input", "2,048 points per modality in 5 R½; new draw and rotation every epoch"],
    ["seeds", "4869, 2, 3 (final); 4869 (controls)"],
    ["compute", `one H100 per run, ${fx.compute.hours_per_run} h, ${n.final_train.gpu_peak_gb.toFixed(1)} GB peak`],
  ], { x: X0, y: 1.5, w: CW, colW: [2.6, CW - 2.6], fontSize: 14, alignFrom: 9 });
  s.addNotes("The final configuration in full.");
}
{
  const s = slide("BACKUP", "Backup", "All Phase C ablations");
  image(s, path.join(FIG, "thesis_fig2_phase_c_ablations.png"), { x: X0, y: 1.35, w: CW, h: 5.5 });
  s.addNotes("Best validation R@1 for every ablation; the shaded band is the ±2 point threshold around the reference.");
}
{
  const s = slide("BACKUP", "Backup", "Retrieval on the test set");
  const rm = T.retrieval_main;
  table(s, rm.header, rm.rows, { x: X0, y: 1.5, w: CW, colW: [3.6, 1.3, 1.3, 1.5, 1.3, 1.5, 1.63], fontSize: 12 });
  const rp = T.retrieval_pairs;
  table(s, rp.header, rp.rows, { x: X0, y: 4.45, w: CW, colW: [3.1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.53], fontSize: 12 });
  s.addNotes("Full retrieval tables: the summary over directions and R@1 per modality pair.");
}
{
  const s = slide("BACKUP", "Backup", "Probes per property");
  image(s, path.join(FIG, "thesis_fig5_probe_properties.png"), { x: X0, y: 1.35, w: CW, h: 5.5 });
  s.addNotes("Test R squared per property for each modality's embedding, and the gain over hand-made statistics.");
}
{
  const s = slide("BACKUP", "Backup", "Dark-matter controls: full numbers");
  const pr = (c30 || dm).probes;
  const props = [["log_mstar_total", "stellar mass"], ["rhalf_star_kpc", "stellar half-mass radius"], ["gas_fraction", "gas fraction"],
    ["log_mdm_total", "dark-matter mass"], ["log_mbh", "black-hole mass"], ["log_z_stars", "stellar metallicity"], ["vmax", "max. circular velocity"]];
  const keys = ["C21-stellar", "c30-subset", "c29-Rvmax", "stats-stellar", "stats-Rvmax"].filter((k) => pr[k]);
  const head = { "C21-stellar": "C21 stellar", "c30-subset": "c30 same galaxies", "c29-Rvmax": "c29 halo", "stats-stellar": "stats stellar", "stats-Rvmax": "stats halo" };
  const cw = keys.length > 4 ? 1.45 : 1.4;
  table(s, [`DM embedding, R² (${(c30 || dm).shared.val} val. galaxies)`, ...keys.map((k) => head[k])],
    [...props.map(([k, l]) => [l, ...keys.map((q) => f2(pr[q].dm.props[k]))]), ["mean of 18", ...keys.map((q) => f2(pr[q].dm.mean_r2))]],
    { x: X0, y: 1.5, w: 3.6 + keys.length * cw, colW: [3.6, ...keys.map(() => cw)], fontSize: 14, boldLast: true });
  s.addNotes(`Kept: ${n.rvmax_subset.kept} of ${n.rvmax_subset.usable} usable galaxies; ${n.rvmax_subset.beyond_extent} have R_vmax beyond the ` +
    `stored particles and ${n.rvmax_subset.too_few_dm} too few dark-matter particles inside it.`);
}
{
  const src = c30 || dm, r = src.retrieval;
  const keys = ["C21-stellar", "c30-subset", "c29-Rvmax"].filter((k) => r[k]);
  const head = { "C21-stellar": "C21: stellar frame, all galaxies", "c30-subset": "c30: stellar frame, same galaxies", "c29-Rvmax": "c29: halo frame" };
  const s = slide("BACKUP", "Backup", "Dark-matter controls: retrieval per pair");
  const pairs = [["stars-dm", "stars – DM"], ["stars-gas", "stars – gas"], ["stars-hi", "stars – HI"], ["dm-gas", "DM – gas"],
    ["dm-hi", "DM – HI"], ["gas-hi", "gas – HI"]];
  table(s, [`R@1 (%), ${src.shared.val} val. galaxies`, ...keys.map((k) => head[k])],
    [...pairs.map(([k, l]) => [l, ...keys.map((q) => f1(r[q].pairs[k]))]),
      ["mean, 10 directions (no gas–HI)", ...keys.map((q) => f1(r[q].r1_no_gas_hi))],
      ["R@10, 10 directions", ...keys.map((q) => f1(r[q].r10_no_gas_hi))]],
    { x: X0, y: 1.5, w: 3.9 + keys.length * 2.7, colW: [3.9, ...keys.map(() => 2.7)], fontSize: 14 });
  s.addNotes("Retrieval on the shared validation galaxies, mean of both directions. Chance is 1 in " + src.shared.val +
    " for pairs with stars and dark matter; gas and HI exist for fewer galaxies, so their galleries are smaller.");
}
{
  const s = slide("BACKUP", "Backup", "Training curves: final setup and controls");
  image(s, path.join(FIG, "control_curves.png"), { x: X0, y: 1.45, w: CW, h: 4.6 });
  text(s, "The final setup is scored on its 520 validation galaxies, the controls on their 428; the like-for-like comparison uses the shared galaxies.",
    { x: X0, y: 6.2, w: CW, h: 0.6, fontSize: 13, color: C.text2 });
  s.addNotes("Validation R@1 during training. The curves show the course of training only, because the validation sets differ.");
}
{
  const s = slide("BACKUP", "Backup", "Code, data and timeline");
  bullets(s, [
    [{ text: "Code: ", options: { bold: true } }, { text: "github.com/Apoptoxin-4869-APTX/GalaxyCLIP-3D (public)" }],
    [{ text: "Data: ", options: { bold: true } }, { text: "/cluster/projects/ska/illustris/tng50-1.pointclouds/v3 (DATASET.md, SELECTION.md)" }],
    [{ text: "Pipeline: ", options: { bold: true } }, { text: "tools/build_selection.py, build_pointclouds.py, finalize_pointclouds.py, check_pointclouds.py" }],
    [{ text: "Model and scoring: ", options: { bold: true } }, { text: "galaxyclip3d/train_clip.py, tools/eval_clip.py, probe_clip.py, compare_probes.py, clip_report.py" }],
    [{ text: "This report and deck: ", options: { bold: true } }, { text: "report/ (collect_numbers.py, figures.py, build_pdf.py, build_deck.js)" }],
  ], { x: X0, y: 1.5, w: 6.3, h: 5.2, fontSize: 14 });
  table(s, ["Date", "Step"], n.timeline, { x: 7.2, y: 1.5, w: 5.53, colW: [1.1, 4.43], fontSize: 12, alignFrom: 9 });
  s.addNotes("Where to find everything, and how the project progressed.");
}

// ---------------------------------------------------------------- write
(async () => {
  await pres.writeFile({ fileName: DECK });
  await applyTheme(DECK, THEME);
  console.log(`wrote ${DECK}`);
})();
