// Builds report/PathSnap_Technical_Report.docx from the result files, tables and figures in this repository.
// Numbers are read from results/ and report/tables/; only the short narrative is typed.
// Run from the repository root: node report/build_report.js
const fs = require("fs");
const path = require("path");
const D = require("docx");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType,
  HeadingLevel, LevelFormat, Footer, PageNumber, BorderStyle } = D;

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "report", "figures");
const TAB = path.join(ROOT, "report", "tables");
const RES = path.join(ROOT, "results");

// ---------------------------------------------------------------- data
function parseCSV(text) {
  const rows = []; let row = [], cur = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { cur += '"'; i++; } else q = false; } else cur += c; }
    else if (c === '"') q = true;
    else if (c === ",") { row.push(cur); cur = ""; }
    else if (c === "\n") { row.push(cur); rows.push(row); row = []; cur = ""; }
    else if (c !== "\r") cur += c;
  }
  if (cur.length || row.length) { row.push(cur); rows.push(row); }
  const head = rows.shift();
  return rows.filter(r => r.length === head.length).map(r => Object.fromEntries(head.map((h, i) => [h, r[i]])));
}
const csv = p => parseCSV(fs.readFileSync(p, "utf8"));
const json = p => JSON.parse(fs.readFileSync(p, "utf8"));
const num = x => (x === "" || x === undefined ? NaN : Number(x));
const pc = (x, d = 1) => (isNaN(x) ? "n/a" : (100 * x).toFixed(d) + "%");
const f = (x, d = 3) => (isNaN(x) ? "n/a" : Number(x).toFixed(d));
const sec = x => (isNaN(x) ? "n/a" : Number(x).toFixed(2) + " s");
const idOf = r => (r.partition !== undefined ? r.partition : (r.trip !== undefined ? r.trip : r[""]));

const T = csv(path.join(TAB, "all_models_table.csv")).map(r => { for (const k of Object.keys(r)) if (!["model", "key", "family", "config"].includes(k)) r[k] = num(r[k]); return r; });
const M = Object.fromEntries(T.map(r => [r.model, r]));
const EVS = csv(path.join(RES, "evaluator_test", "evaluator_test_summary.csv"));
const STS = csv(path.join(RES, "student_test", "student_test_summary.csv"));
const PT = csv(path.join(RES, "evaluator_test", "evaluator_test_per_trip_top1.csv"));
const LH = csv(path.join(RES, "lightgbm_hmm", "scores.csv"));
const PART = csv(path.join(TAB, "partition_summary.csv"));
const TUNE = path.join(RES, "tuning");
const tl = json(path.join(TUNE, "lightgbm_tuning_summary.json")), tcv = json(path.join(TUNE, "lightgbm_cv_comparison.json"));
const hmmT = json(path.join(ROOT, "xg boost", "results", "hmm_tuned.json"));
const hmmCfg = json(path.join(ROOT, "xg boost", "models", "hmm_config.json"));
const mlpM = json(path.join(ROOT, "mlp", "results", "mlp_complete_report_metrics.json"));
const ev = (m, s) => EVS.find(r => r.model === m && r.stage === s);
const st = (m, s) => STS.find(r => r.model === m && r.stage === s);
const CL = "classifier only", HM = "classifier + HMM/Viterbi";
const part = (name, col) => num(PART.find(r => idOf(r) === name)[col]);
const LG = M["LightGBM"], RF = M["Random Forest"], ML = M["Deep MLP"], LRG = M["Logistic Regression"], XG = M["XGBoost"];
const lhU = (sp, s) => LH.find(r => r.split === sp && r.model === "LightGBM (untuned)" && r.stage === s);
const lhT = (sp, s) => LH.find(r => r.split === sp && r.model === "LightGBM (tuned)" && r.stage === s);
const hurt = PT.map(r => ({ t: idOf(r), c: num(r["LightGBM"]), h: num(r["LightGBM + HMM"]) })).filter(r => r.h < r.c).map(r => r.t);

// ---------------------------------------------------------------- minimal layout helpers
const FONT = "Arial", INK = "1A1A1A", GREY = "6B6B6B", RULE = "D4D4D4";
const run = (t, o = {}) => new TextRun({ text: t, font: FONT, size: o.size || 20, bold: o.bold, color: o.color || INK });
const P = (t, o = {}) => new Paragraph({ children: [run(t, o)], spacing: { after: o.after ?? 100, line: 276 }, keepNext: o.keepNext });
const H1 = (t, brk = true) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: brk, children: [run(t, { size: 30, bold: true })] });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [run(t, { size: 23, bold: true })] });
const L = t => new Paragraph({ numbering: { reference: "list", level: 0 }, children: [run(t)], spacing: { after: 50, line: 270 } });
let figN = 0, tabN = 0;
const dims = file => { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; };
function Fig(name, caption, maxW = 600) {
  const file = path.join(FIG, name); const { w, h } = dims(file); const sc = Math.min(1, maxW / w); figN++;
  return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 40 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: Math.round(w * sc), height: Math.round(h * sc) }, altText: { title: name, description: caption, name } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 180 }, children: [run(`Figure ${figN}. ${caption}`, { size: 17, color: GREY })] })];
}
const line = { style: BorderStyle.SINGLE, size: 4, color: RULE };
const borders = { top: line, bottom: line, left: line, right: line };
function Tbl(headers, rows, widths, o = {}) {
  const sz = o.size || 17, total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, head) => new TableCell({ borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" } : undefined, margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new Paragraph({ alignment: i === 0 || o.left ? AlignmentType.LEFT : AlignmentType.CENTER, children: [run(String(t), { size: sz, bold: head })] })] });
  tabN++;
  return [new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
      rows: [headers ? new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }) : null, ...rows.map(r => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false)) }))].filter(Boolean) }),
    new Paragraph({ spacing: { before: 50, after: 180 }, children: [run(o.caption ? `Table ${tabN}. ${o.caption}` : "", { size: 17, color: GREY })] })];
}

// ---------------------------------------------------------------- content
const C = [];
// title page
C.push(new Paragraph({ spacing: { before: 3200, after: 120 }, children: [run("PathSnap", { size: 60, bold: true })] }));
C.push(new Paragraph({ spacing: { after: 100 }, children: [run("Distributed Map-Matching and Route Reconstruction", { size: 30 })] }));
C.push(new Paragraph({ spacing: { after: 700 }, children: [run("Technical report", { size: 24, color: GREY })] }));
C.push(P("UE24CS352A: Machine Learning, PES University", { color: GREY, after: 40 }));
C.push(P("Anirudh Ramesh, Kedar S Vastrad, Sanjeev Murthy, sanjithks15-cyber", { color: GREY, after: 40 }));
C.push(P("github.com/CloudStone16/ML-Hackathon", { color: GREY, after: 40 }));
C.push(P("18 November 2026", { color: GREY }));

// ================= 1 EDA
C.push(H1("1 Dataset understanding and exploratory data analysis"));
C.push(H2("1.1 How did you preprocess the data?"));
[`Load each trip and clean the track: drop invalid rows, sort by time, remove duplicate timestamps. No row had to be removed in any of the 100 trips.`,
 `Project longitude and latitude to metres around each trip's own centroid, because the trips come from many countries.`,
 `Compute speed and headings (windows of 2, 5 and 10 points) from the track.`,
 `Align each GPS point to the reference route with a monotone dynamic programme to get a per-point label. Points more than 60 m from the aligned arc are not used for training.`,
 `Exclude two training trips (00000033 and 00000054) whose route does not match their GPS.`,
 `Generate candidates: all arcs within 50 m of a point, at most 40 nearest.`,
 `Compute 36 features per (point, arc) pair and split strictly by trip. Scalers and imputers are fitted on training data only.`].forEach(t => C.push(L(t)));
C.push(H2("1.2 Patterns observed, features extracted and why"));
C.push(L("GPS points are close to their road (median 4.6 m to the aligned arc), but the nearest arc is the true one only 54% of the time on student_test."));
C.push(L("Most streets have a reverse twin arc at the same location, so direction decides many cases."));
C.push(L(`Sampling is nominally 1 Hz (96.5% of training steps), with a few long gaps (largest 153,064 s in the training partition).`));
C.push(L("The reference route is a reference, not truth: two training trips do not match their GPS at all."));
C.push(...Tbl(["Feature group", "Features", "Why"], [
  ["Geometry", "distance to arc, position along arc, arc length", "How well the point fits the arc"],
  ["Competition", "distance rank, gap to nearest candidate, number of candidates", "The nearest arc is often wrong"],
  ["Heading", "heading vs arc direction at 4 window sizes", "Separates reverse twins and crossing streets"],
  ["Motion", "speed, time gaps", "Stopped or gappy points have unreliable heading"],
  ["Reverse twin", "twin exists, twin is a candidate, distance difference", "Direction on two-way streets"],
  ["Network", "node degrees at the arc ends", "Junction structure"],
  ["Temporal context", "same arc's distance at neighbouring points, progress, connectivity", "Adds sequence information"]], [1900, 4600, 3138], { left: true, caption: "The 36 features by group (partition: training)." }));
C.push(H2("1.3 Missing values, trajectories and the road network"));
C.push(...Tbl(["Partition", "Trips", "GPS points", "Mean nodes", "Mean arcs", "Steps at 1 s"],
  ["train", "student_test", "evaluator_test"].map(p => [p, part(p, "trips"), part(p, "gps_points").toLocaleString(), Math.round(part(p, "mean_nodes")).toLocaleString(), Math.round(part(p, "mean_arcs")).toLocaleString(), part(p, "mean_pct_1hz").toFixed(1) + "%"]),
  [2200, 1000, 1700, 1600, 1600, 1538], { caption: "Dataset inventory." }));
C.push(...Fig("eda_04_missing_value_checks.png", "Missing-value and validity checks (partitions: training, student_test, evaluator_test). All counts are zero.", 520));
C.push(...Fig("eda_02_trip_locations.png", "Trip centroids (all three partitions)."));
C.push(...Fig("eda_03_sampling_gaps.png", "Sampling intervals and gaps (all three partitions)."));
C.push(...Fig("eda_06_trajectories.png", "GPS trajectories with reference route and road network (partitions: training, student_test).", 540));
C.push(...Fig("eda_07_road_network.png", "Road network around training trip 00000001, with a zoom on parallel and crossing streets."));
C.push(...Fig("eda_05_network_connectivity.png", "Road-network size and connectivity (all three partitions). There are no isolated nodes."));
C.push(...Fig("eda_08_label_quality.png", "Agreement between GPS and reference route (partition: training).", 560));
C.push(...Fig("folium_eda_train_00000046.png", "Folium map of training trip 00000046: GPS (blue), road network (grey), reference route (green)."));

// ================= 2 Models
C.push(H1("2 Model experimentation and comparison"));
C.push(P("Five models were trained with 5-fold GroupKFold cross-validation on the training trips and compared on student_test. Top-1 is observation level, and a point whose true arc is not in the candidate set counts as wrong."));
C.push(H2("2.1 Overall comparison"));
const MODS = ["LightGBM", "Random Forest", "Deep MLP", "Logistic Regression", "XGBoost"];
C.push(...Tbl(["Model", "Family", "Key configuration", "CV macro F1", "CV Top-1", "Student Top-1", "Route continuity", "Student macro F1", "Log loss", "Inference"],
  MODS.map(n => { const m = M[n]; return [n, m.family, m.config, f(m.cv_macro_f1), pc(m.cv_top1), pc(m.test_top1), pc(m.test_continuity), f(m.test_macro_f1), f(m.test_logloss), sec(m.infer_s)]; }),
  [1150, 950, 1500, 700, 750, 800, 950, 800, 700, 1338], { size: 14, caption: "Classifier-only comparison. Inference time is for the 19,160 student_test points." }));
C.push(H2("2.2 Per-model answers"));
const fit = m => `train ${isNaN(m.train_top1) ? "not recorded" : pc(m.train_top1)}, CV ${pc(m.cv_top1)}, student_test ${pc(m.test_top1)}, evaluator_test ${pc(m.eval_top1)}`;
function modelTable(name, o) {
  const m = M[name];
  return [P(name, { bold: true, after: 40, keepNext: true }), ...Tbl(null, [
    ["Why suitable", o.why], ["Strengths and limitations", o.sl], ["Hyperparameters (initial values)", o.hp], ["Loss or objective", o.loss],
    ["Student-test performance", `Top-1 ${pc(m.test_top1)}, macro F1 ${f(m.test_macro_f1)}, log loss ${f(m.test_logloss)}`],
    ["Over or underfitting", `${fit(m)}. ${o.fit}`]], [2400, 7238], { left: true })];
}
C.push(...modelTable("LightGBM", { why: "Boosted trees handle mixed-scale features and missing headings and fit interactions such as distance gap with heading agreement.",
  sl: "Accurate, well calibrated, fast and small (2.7 MB). Scores each point on its own, so it needs the HMM for connected routes.",
  hp: "400 trees, learning rate 0.05, 63 leaves, min child samples 50, subsample 0.8, colsample 0.8, lambda 1.0.", loss: "Binary cross-entropy.",
  fit: "Mild overfitting (train is higher than CV), but CV and held-out scores agree within about 2 points." }));
C.push(...modelTable("Random Forest", { why: "Bagged trees are a robust baseline for non-linear tabular data.",
  sl: "Stable, but uses 9 features, 6 candidates per point and one fixed projection for all trips, which limits accuracy. Poorly calibrated probabilities.",
  hp: "150 trees, depth 15, min split 5, min leaf 2, sqrt features, balanced class weights.", loss: "Gini impurity with class weights.",
  fit: "CV folds range from 75.5% to 98.2%, so it is sensitive to the trip. It is closer to underfitting the heading information than to overfitting." }));
C.push(...modelTable("Deep MLP", { why: "Candidate scoring depends on smooth quantities (distance decay, heading cosines), and the network compares all candidates of a point and neighbouring time steps.",
  sl: "Best classifier-only accuracy on both test sets, but keeps only 5 candidates per point (1.4% misses) and varies more across folds.",
  hp: `Initial: hidden 96, head 48, dropout 0.15, learning rate 0.0025, listwise weight 0.45, positive weight 1.60. Final: hidden ${mlpM.best_hyperparameters.hidden_dim}, head ${mlpM.best_hyperparameters.head_dim}, dropout ${mlpM.best_hyperparameters.dropout}.`,
  loss: "Binary cross-entropy blended with a listwise softmax cross-entropy.",
  fit: "No overfitting: training accuracy is below every held-out score, and validation loss stays below training loss." }));
C.push(...modelTable("Logistic Regression", { why: "The simplest probabilistic baseline for the engineered features.",
  sl: "Fast and stable, but a linear boundary cannot express interactions between distance and heading.",
  hp: "C 1.0, lbfgs, median imputation and standard scaling.", loss: "L2-regularised cross-entropy.",
  fit: "No gap between train, CV and test, so it underfits relative to boosting." }));
C.push(...modelTable("XGBoost", { why: "Boosted trees on a smaller feature set, as a second boosting model.",
  sl: "Fast, and accurate with the HMM, but it has 15 features and only one heading, so classifier-only accuracy drops on evaluator_test.",
  hp: "Initial: 300 trees, depth 6, learning rate 0.1, subsample 0.8, colsample 0.8.", loss: "Logistic loss.",
  fit: "Small train-to-CV gap. The weakness is the feature set, and the HMM recovers most of it." }));
C.push(H2("2.3 Model visualisations"));
C.push(...Fig("model_01_cv_comparison.png", "5-fold cross-validation macro F1 and Top-1 (partition: training).", 580));
C.push(...Fig("model_02_student_test.png", "Student-test Top-1 and macro F1."));
C.push(...Fig("model_03_log_loss.png", "Student-test log loss.", 480));
C.push(...Fig("model_05_confusion_matrices.png", "Candidate-level confusion matrices on student_test.", 540));
C.push(...Fig("model_06_training_curves.png", "Training and validation loss. Only the Deep MLP and XGBoost produce training histories.", 600));
C.push(...Fig("model_04_evaluator_test.png", "Evaluator-test Top-1 and route continuity, classifier only and with HMM (final check).", 580));
C.push(H2("2.4 Model selection"));
C.push(P("The final model was chosen on student_test and cross-validation only (0.6 and 0.4 weights). evaluator_test is a final check. LightGBM with HMM ranks first, with a candidate miss rate of 0.14% against 1.42% for the Deep MLP, better calibrated probabilities and a saved model file that is exactly what was evaluated. The top two pipelines are close, and the Deep MLP is ahead by 0.3 points on evaluator_test."));
C.push(...Fig("model_10_selection.png", "Selection score (bars) and the evaluator_test Top-1 (dots).", 560));

// ================= 3 Bayesian
C.push(H1("3 Bayesian hyperparameter optimisation (LightGBM)"));
const U = (sp, s) => lhU(sp, s), Tn = (sp, s) => lhT(sp, s);
C.push(...Tbl(null, [
  ["1. Hyperparameters searched", "n_estimators, learning_rate, num_leaves, min_child_samples, subsample, colsample_bytree, reg_lambda, reg_alpha, min_split_gain"],
  ["2. Ranges", "n_estimators 150 to 800, learning_rate 0.02 to 0.15 (log), num_leaves 15 to 255 (log), min_child_samples 10 to 300 (log), subsample 0.5 to 1.0, colsample_bytree 0.4 to 1.0, reg_lambda 0.001 to 30 (log), reg_alpha 0.001 to 10 (log), min_split_gain 0 to 1"],
  ["3. Objective", "Mean observation-level Top-1 over 5-fold GroupKFold on the training trips. No test data is used."],
  ["4. Trials", `${tl.n_trials} (Optuna TPE with a median pruner): 19 completed, 11 pruned.`],
  ["5. Best values", `n_estimators ${tl.best_params.n_estimators}, learning_rate ${tl.best_params.learning_rate.toFixed(4)}, num_leaves ${tl.best_params.num_leaves}, min_child_samples ${tl.best_params.min_child_samples}, subsample ${tl.best_params.subsample.toFixed(2)}, colsample_bytree ${tl.best_params.colsample_bytree.toFixed(2)}, reg_lambda ${tl.best_params.reg_lambda.toFixed(4)}, reg_alpha ${tl.best_params.reg_alpha.toFixed(4)}, min_split_gain ${tl.best_params.min_split_gain.toFixed(2)}`],
  ["6. Did tuning help?", `Only slightly in cross-validation (+${(100 * (tcv.tuned.top1_acc.mean - tcv.untuned.top1_acc.mean)).toFixed(2)} points). On the test sets it is level, and with the HMM it is worse, so the untuned model is kept.`]],
  [2400, 7238], { left: true, caption: "Bayesian optimisation setup and outcome." }));
C.push(...Tbl(["Metric", "Untuned", "Tuned"], [
  ["CV macro F1", f(tcv.untuned.cand_macro_f1.mean), f(tcv.tuned.cand_macro_f1.mean)],
  ["Student-test Top-1", pc(num(U("student_test", CL).top1_acc), 2), pc(num(Tn("student_test", CL).top1_acc), 2)],
  ["Student-test macro F1", f(num(U("student_test", CL).macro_f1)), f(num(Tn("student_test", CL).macro_f1))],
  ["Student-test log loss", f(num(U("student_test", CL).log_loss)), f(num(Tn("student_test", CL).log_loss))],
  ["Inference time (student_test)", sec(tl.student_test_untuned.inference_s), sec(tl.student_test_tuned.inference_s)]], [4400, 2600, 2638], { caption: "Untuned and tuned LightGBM, same metrics and partitions." }));
C.push(...Fig("tune_01_trial_history.png", "Optimisation trials for LightGBM, XGBoost and the Deep MLP (partition: training).", 600));
C.push(...Fig("tune_03_untuned_vs_tuned_trees.png", "CV and student-test Top-1 before and after tuning.", 560));

// ================= 4 HMM
C.push(H1("4 HMM and Viterbi route reconstruction"));
C.push(...Tbl(null, [
  ["1. Hidden states", "The candidate arcs of each GPS point (up to 40), indexed so that each state maps back to a road segment id."],
  ["2. Emission", `alpha times the log of the classifier probability of the candidate, with alpha ${hmmCfg.params.alpha.toFixed(2)}.`],
  ["3. Transition", `Based on the number of road steps h between consecutive candidate arcs (search up to ${hmmCfg.maxhops} hops): same or next arc costs 0, skipping arcs costs ${hmmCfg.params.lam.toFixed(2)} per skipped arc, a U-turn onto the reverse arc costs ${hmmCfg.params.uturn.toFixed(1)}.`],
  ["4. Missing candidates and gaps", `Points without candidates are skipped. A gap of more than ${hmmCfg.params.gap_s.toFixed(1)} s restarts the chain.`],
  ["5. Disconnected arcs", `A finite penalty of ${hmmCfg.params.disc.toFixed(1)} applies, so the chain can recover but breaks need strong evidence.`],
  ["6. Did it improve the route?", `Yes. See the table and figure below. The decoder parameters were tuned on out-of-fold training probabilities (${pc(hmmT.tuning.best_value, 1)} Top-1), not on any test set.`]],
  [2400, 7238], { left: true, caption: "HMM design (Viterbi recurrence: delta_t(j) = emit_t(j) + max_i [delta_(t-1)(i) + trans(i, j)])." }));
C.push(...Tbl(["Pipeline", "Partition", "Top-1, classifier only", "Top-1 with HMM", "Continuity, classifier only", "Continuity with HMM"],
  ["LightGBM", "Deep MLP", "XGBoost"].flatMap(m => [["student_test", STS], ["evaluator_test", EVS]].map(([sp, S]) => { const a = S.find(r => r.model === m && r.stage === CL), b = S.find(r => r.model === m && r.stage === HM); return [m, sp, pc(num(a.top1_acc)), pc(num(b.top1_acc)), pc(num(a.route_continuity)), pc(num(b.route_continuity))]; })),
  [1700, 1500, 1600, 1500, 1700, 1638], { caption: "Classifier only against classifier plus HMM." }));
C.push(...Fig("hmm_01_classifier_vs_hmm.png", "Top-1 and route continuity with and without HMM decoding.", 600));
C.push(P(`The HMM did not help on every trip: for LightGBM it lowered Top-1 on ${hurt.length} of 10 evaluator trips (${hurt.join(", ")}), where the classifier was wrong for a long stretch and the transition costs reinforced the wrong arc.`));

// ================= 5 Folium
C.push(H1("5 Geographical visualisation using Folium"));
C.push(P("Maps for evaluator_test trips: raw GPS (blue), road network (grey), reference route (green), predicted route from LightGBM with HMM (red) and points the classifier alone got wrong (orange). Interactive versions are in report/maps."));
C.push(...Fig("folium_00000030.png", "Evaluator trip 00000030 (hardest trip).", 600));
C.push(...Fig("folium_00000015.png", "Evaluator trip 00000015 (largest HMM gain).", 600));
C.push(...Fig("map_01_route_comparison_static.png", "Reference route, classifier-only route and HMM route for the same trip.", 600));

// ================= Screenshots
C.push(H1("Screenshots"));
C.push(P("[UI screenshot]", { color: GREY }));
C.push(P("[API response screenshot, terminal of the receiving laptop]", { color: GREY }));

// ---------------------------------------------------------------- document
const doc = new Document({
  creator: "PathSnap team", title: "PathSnap technical report",
  styles: { default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 30, bold: true, color: INK }, paragraph: { spacing: { before: 120, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 23, bold: true, color: INK }, paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } }] },
  numbering: { config: [{ reference: "list", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 250 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, right: 1134, bottom: 1134, left: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: GREY })] })] }) },
    children: C }] });
Packer.toBuffer(doc).then(buf => { const out = path.join(ROOT, "report", "PathSnap_Technical_Report.docx"); fs.writeFileSync(out, buf); console.log("wrote", out, (buf.length / 1e6).toFixed(1) + " MB", "figures:", figN, "tables:", tabN); });
