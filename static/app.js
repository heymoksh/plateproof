// PlateProof UI. All server text is inserted with textContent (never innerHTML),
// because filenames, order details and metadata can be controlled by the uploader.

const LEVELS = {
  HIGH_REVIEW_PRIORITY: { label: "High review priority", colour: "var(--high)" },
  REVIEW_RECOMMENDED: { label: "Review recommended", colour: "var(--review)" },
  NO_SIGNIFICANT_INDICATORS: { label: "No significant indicators", colour: "var(--clear)" },
  INSUFFICIENT_EVIDENCE: { label: "Insufficient evidence", colour: "var(--insufficient)" },
};

const FIELDS = ["order_id", "customer_id", "restaurant", "ordered_items", "order_placed_at", "delivered_at",
  "complaint_type", "image_source", "description"];

const EXAMPLES = [
  { label: "Genuine photo", files: ["1_camera_original.jpg"],
    fields: { order_id: "ORD-1001", order_placed_at: "2026-05-02T16:05", delivered_at: "2026-05-02T16:35", complaint_type: "spilled_or_damaged" } },
  { label: "Edited after capture", files: ["2_edited_after_capture.jpg"], fields: { order_id: "ORD-1002", complaint_type: "foreign_object" } },
  { label: "AI-generated", files: ["3_ai_generated.png"], fields: { order_id: "ORD-1003", complaint_type: "quality_issue" } },
  { label: "Old photo", files: ["5_taken_before_order.jpg"], fields: { order_id: "ORD-1005", order_placed_at: "2026-05-02T19:00" } },
  { label: "Two different phones", files: ["1_camera_original.jpg", "5_taken_before_order.jpg"], fields: { order_id: "ORD-1006" } },
  { label: "Reused photo", files: ["4_messaging_app_copy.jpg"], fields: { order_id: "ORD-1004" },
    note: "Shows a reuse match if the Genuine photo example was checked before." },
];

const form = document.getElementById("claim-form");
const input = document.getElementById("images");
const drop = document.getElementById("drop");
const thumbs = document.getElementById("thumbs");
const results = document.getElementById("results");
const submitBtn = document.getElementById("submit");
const formError = document.getElementById("form-error");
let maxPhotos = 6;
let selected = []; // [{file, url}]

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "style") node.style.cssText = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const formatBytes = (n) => (n == null ? "" : n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1048576).toFixed(1)} MB`);
const human = (s) => String(s).replaceAll("_", " ");

function showError(message) {
  formError.textContent = message;
  formError.hidden = !message;
}

// ---------- intake ----------

async function loadHealth() {
  const box = document.getElementById("use-vision");
  const note = document.getElementById("vision-note");
  try {
    const health = await (await fetch("/api/health")).json();
    maxPhotos = health.max_photos || maxPhotos;
    document.getElementById("max-mb").textContent = health.max_upload_mb;
    document.getElementById("max-photos").textContent = maxPhotos;
    if (health.vision_configured) {
      box.checked = true;
      note.textContent = `(sends a reduced copy of each photo to Google, model ${health.vision_model})`;
    } else {
      box.disabled = true;
      document.getElementById("vision-option").classList.add("disabled");
      note.textContent = "(not configured: add GEMINI_API_KEY to .env)";
    }
  } catch {
    box.disabled = true;
    note.textContent = "(status unknown)";
  }
}

function renderThumbs() {
  thumbs.replaceChildren(...selected.map((item, i) => {
    const img = el("img", { src: item.url, alt: item.file.name });
    const li = el("li", { class: "thumb" }, img, el("span", { class: "n" }, i + 1),
      el("button", { type: "button", "aria-label": `Remove ${item.file.name}`, onclick: () => removeFile(i) }, "×"));
    img.addEventListener("error", () => img.replaceWith(el("div", { class: "noimg" }, item.file.name)));
    return li;
  }));
}

function addFiles(files) {
  showError("");
  for (const file of files) {
    if (selected.length >= maxPhotos) {
      showError(`A claim can have at most ${maxPhotos} photos.`);
      break;
    }
    selected.push({ file, url: URL.createObjectURL(file) });
  }
  renderThumbs();
}

function removeFile(index) {
  URL.revokeObjectURL(selected[index].url);
  selected.splice(index, 1);
  renderThumbs();
}

function clearFiles() {
  selected.forEach((item) => URL.revokeObjectURL(item.url));
  selected = [];
  renderThumbs();
}

input.addEventListener("change", () => { addFiles([...input.files]); input.value = ""; });
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("dragging"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("dragging")));
drop.addEventListener("drop", (e) => { e.preventDefault(); addFiles([...e.dataTransfer.files]); });

async function loadExample(example) {
  showError("");
  clearFiles();
  for (const name of FIELDS) form.elements[name].value = form.elements[name].tagName === "SELECT" ? "unknown" : "";
  for (const [name, value] of Object.entries(example.fields)) form.elements[name].value = value;
  document.getElementById("order-details").open = true;
  try {
    const files = await Promise.all(example.files.map(async (name) => {
      const response = await fetch(`/samples/${name}`);
      if (!response.ok) throw new Error(name);
      const blob = await response.blob();
      return new File([blob], name, { type: blob.type || "image/jpeg" });
    }));
    addFiles(files);
    if (example.note) showError(example.note);
  } catch {
    showError("The example photos could not be loaded. Check that the samples folder exists.");
  }
}

document.getElementById("examples").replaceChildren(...EXAMPLES.map((ex) =>
  el("button", { type: "button", onclick: () => loadExample(ex) }, ex.label)));

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("");
  if (!selected.length) {
    showError("Add at least one photo first.");
    return;
  }
  const data = new FormData();
  selected.forEach((item) => data.append("images", item.file, item.file.name));
  for (const name of FIELDS) {
    const value = form.elements[name].value.trim();
    if (value && value !== "unknown") data.append(name, value);
  }
  data.append("save_to_history", form.elements.save_to_history.checked ? "true" : "false");
  data.append("use_vision", form.elements.use_vision.checked ? "true" : "false");

  submitBtn.disabled = true;
  submitBtn.textContent = "Checking…";
  results.replaceChildren(el("div", { class: "panel loading" }, el("div", { class: "spinner" }), `Checking ${selected.length} photo${selected.length === 1 ? "" : "s"}…`));
  try {
    const response = await fetch("/api/analyze", { method: "POST", body: data });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      const message = body?.error?.message || `The server returned an error (${response.status}).`;
      showError(message);
      results.replaceChildren(el("div", { class: "empty" }, el("h2", {}, "The claim could not be checked"), el("p", {}, message)));
      return;
    }
    renderClaim(body, selected.map((s) => s));
  } catch {
    const message = "Could not reach the server. Check that the application is still running.";
    showError(message);
    results.replaceChildren(el("div", { class: "empty" }, el("h2", {}, "Connection problem"), el("p", {}, message)));
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Check photos";
  }
});

// ---------- report ----------

function section(title, lead, ...content) {
  return el("section", { class: "panel section" }, el("h3", {}, title), lead ? el("p", { class: "lead" }, lead) : null, ...content);
}

function ledger(claim) {
  const photos = claim.photos;
  const best = photos.reduce((b, p, i) => (p.risk.score > photos[b].risk.score ? i : b), 0);
  const tag = photos.length > 1 ? `Photo ${best + 1}: ` : "";
  const items = [
    ...photos[best].evidence.filter((e) => e.points > 0).map((e) => ({ ...e, title: tag + e.title })),
    ...claim.claim_evidence.filter((e) => e.points > 0).map((e) => ({ ...e, title: `All photos: ${e.title}` })),
  ];
  let used = 0;
  const segments = [];
  for (const item of items) {
    const width = Math.min(item.points, 100 - used);
    if (width <= 0) break;
    used += width;
    segments.push(el("div", { class: "ledger-seg", style: `width:${width}%`, title: `${item.title}: +${item.points}` }, width >= 8 ? `+${item.points}` : ""));
  }
  const tick = (v, label) => el("div", { class: `ledger-tick${v === 0 ? " start" : v === 100 ? " end" : ""}`, style: `left:${v}%` }, label);
  const note = photos.length > 1 ? " With several photos, the score is the highest single-photo score plus cross-photo points." : "";
  return el("div", { class: "ledger" },
    el("div", { class: "ledger-bar", role: "img", "aria-label": `Risk score ${claim.risk.score} out of 100` }, segments),
    el("div", { class: "ledger-scale" },
      el("div", { class: "ledger-mark", style: "left:20%" }), el("div", { class: "ledger-mark", style: "left:50%" }),
      tick(0, "0"), tick(20, "20 review"), tick(50, "50 high priority"), tick(100, "100")),
    el("p", { class: "ledger-caption" }, `Risk score ${claim.risk.score} of 100. ${claim.risk.score_note}${note}`));
}

function verdict(claim) {
  const level = LEVELS[claim.risk.level];
  const n = claim.photos.length;
  return el("section", { class: "panel verdict", style: `--level:${level.colour}` },
    el("p", { class: "level" }, level.label),
    el("p", { class: "summary" }, claim.risk.summary),
    el("p", { class: "recommendation" }, claim.risk.recommendation),
    ledger(claim),
    el("p", { class: "meta" }, `Case ${claim.case_id}, ${n} photo${n === 1 ? "" : "s"}, checked ${new Date(claim.analyzed_at).toLocaleString()}`));
}

function indicatorList(items) {
  return el("ul", { class: "indicators" }, items.map(({ e, source }) => el("li", { class: "indicator" },
    el("div", { class: `points ${e.severity}` }, `+${e.points}`),
    el("div", {}, el("h4", {}, source ? `${source}: ${e.title}` : e.title), el("p", {}, e.detail),
      e.caveat ? el("p", { class: "caveat" }, e.caveat) : null))));
}

function whySection(claim) {
  const multi = claim.photos.length > 1;
  const items = [
    ...claim.claim_evidence.filter((e) => e.points > 0).map((e) => ({ e, source: "All photos" })),
    ...claim.photos.flatMap((p, i) => p.evidence.filter((e) => e.points > 0).map((e) => ({ e, source: multi ? `Photo ${i + 1}` : null }))),
  ];
  if (!items.length) {
    const text = claim.risk.level === "INSUFFICIENT_EVIDENCE"
      ? "No indicators were found, but the photos lack the metadata needed for most checks, so this is not a clean result."
      : "No indicators were found by the automated checks.";
    return section("Why this result", null, el("p", {}, text));
  }
  return section("Why this result", "Each indicator adds hand-set points. Check each one before acting on it.", indicatorList(items));
}

function acrossSection(claim) {
  const info = claim.claim_evidence.filter((e) => e.points === 0);
  if (claim.photos.length < 2 || !info.length) return null;
  return section("Across photos", "How the claim's photos compare with each other.",
    el("ul", { class: "observations" }, info.map((e) => el("li", {}, el("strong", {}, e.title), ". ", e.detail))));
}

function limitationsSection(claim) {
  const multi = claim.photos.length > 1;
  const items = [...claim.limitations];
  claim.photos.forEach((p, i) => p.limitations.forEach((l) => items.push(multi ? `Photo ${i + 1}: ${l}` : l)));
  const unique = [...new Set(items)];
  if (!unique.length) return null;
  return section("Limitations of this result", null, el("ul", { class: "limitations" }, unique.map((l) => el("li", {}, l))));
}

function rejectedSection(claim) {
  if (!claim.rejected.length) return null;
  return section("Photos that could not be checked", null,
    el("ul", { class: "limitations rejected" }, claim.rejected.map((r) => el("li", {}, `${r.filename || "unnamed"}: ${r.message}`))));
}

function photoPanel(photo, index, localUrl, showVerdict = true) {
  const level = LEVELS[photo.risk.level];
  const meta = photo.metadata || {};
  const img = photo.image || {};
  const integrity = img.integrity || {};
  const preview = integrity.embedded_preview || {};
  const info = photo.evidence.filter((e) => e.points === 0 && !e.advisory);
  const advisory = photo.evidence.filter((e) => e.advisory);
  const rows = [
    ["File", `${photo.filename || "unnamed"} (${img.format}, ${formatBytes(img.file_size_bytes)})`],
    ["Dimensions", `${img.width} x ${img.height} pixels (${img.megapixels} MP)`],
    ["Camera", [meta.camera?.make, meta.camera?.model].filter(Boolean).join(" ") || "Not recorded"],
    ["Captured", meta.timestamps?.captured || "Not recorded"],
    ["Last modified", meta.timestamps?.modified || "Not recorded"],
    ["Software", meta.software || "Not recorded"],
    ["GPS", meta.gps ? `${meta.gps.latitude}, ${meta.gps.longitude}` : "Not recorded"],
    ["Embedded preview", !preview.present ? "None" : preview.compared ? `${human(preview.result)} (local difference ${preview.local_difference ?? "n/a"})` : "Present, not compared"],
    ["JPEG quality (est.)", integrity.jpeg_quality_estimate ?? "Not a JPEG"],
    ["Content Credentials", meta.content_credentials?.present ? "Manifest detected (not verified)" : "None"],
    ["SHA-256", photo.hashes?.sha256 || ""],
    ["Perceptual hashes", `pHash ${photo.hashes?.phash || ""}, dHash ${photo.hashes?.dhash || ""}`],
  ];

  const figures = [];
  if (localUrl) figures.push(el("figure", {}, el("img", { src: localUrl, alt: `Photo ${index + 1}`, onerror: (e) => e.target.closest("figure").remove() }), el("figcaption", {}, "Photo as uploaded")));
  if (photo.ela_preview) figures.push(el("figure", {}, el("img", { src: photo.ela_preview, alt: "Error level analysis" }),
    el("figcaption", {}, "Error level analysis: a visual aid only. Bright areas are common at edges and textures and are not proof of editing.")));

  const vision = photo.vision?.status === "completed" ? photo.vision : null;
  const ml = photo.ml_detector?.status === "completed" ? photo.ml_detector : null;
  const o = vision?.observations;

  return el("div", { class: "photo-panel", role: showVerdict ? "tabpanel" : null, id: `photo-${index}` },
    showVerdict ? el("section", { class: "panel section photo-verdict", style: `--level:${level.colour}` },
      el("p", { class: "level" }, `${level.label} (score ${photo.risk.score})`),
      el("p", {}, photo.risk.summary)) : null,
    info.length ? section("Observations", "Context that does not add to the score.",
      el("ul", { class: "observations" }, info.map((e) => el("li", {}, el("strong", {}, e.title), ". ", e.detail)))) : null,
    vision || ml || advisory.length ? el("section", { class: "panel section advisory" }, el("h3", {}, "Model observations (advisory)"),
      el("p", { class: "lead" }, "Produced by AI models. Shown for the reviewer's judgement and never used in the score."),
      advisory.length ? el("ul", { class: "observations" }, advisory.map((e) => el("li", {}, el("strong", {}, e.title), ". ", e.detail))) : null,
      o ? el("dl", { class: "facts", style: "margin-top:10px" },
        el("dt", {}, "Scene"), el("dd", {}, human(o.scene_type)),
        el("dt", {}, "Image type"), el("dd", {}, human(o.image_kind)),
        el("dt", {}, "Items"), el("dd", {}, o.items_visible.join(", ") || "None listed"),
        el("dt", {}, "Problem visible"), el("dd", {}, `${o.issue_visible}${o.issue_description ? `: ${o.issue_description}` : ""}`),
        el("dt", {}, "Matches claim"), el("dd", {}, `${human(o.consistency_with_claim)}${o.consistency_explanation ? `: ${o.consistency_explanation}` : ""}`),
        o.visible_text.length ? [el("dt", {}, "Visible text"), el("dd", {}, o.visible_text.join(" | "))] : null) : null,
      ml ? el("p", { style: "margin-top:10px" }, `Experimental classifier output ${ml.ai_generated_score} for "${ml.positive_class}" (0 to 1, uncalibrated). ${ml.note}`) : null) : null,
    section("What was checked", null, el("ul", { class: "checks" }, photo.checks.map((c) =>
      el("li", {}, el("span", { class: `status ${c.status}` }, c.status), el("span", {}, c.label), el("span", { class: "msg" }, c.message))))),
    section("Photo details", null,
      figures.length ? el("div", { class: "images" }, figures) : null,
      el("dl", { class: "facts", style: "margin-top:16px" }, rows.map(([k, v]) => [el("dt", {}, k), el("dd", { class: k.includes("SHA") || k.includes("hashes") ? "mono" : null }, String(v))])),
      photo.duplicates.length ? el("div", { style: "margin-top:16px" }, el("h4", {}, "Matching earlier submissions"),
        el("ul", { class: "observations" }, photo.duplicates.map((m) => el("li", {},
          `${m.match} match: case ${m.case_id}, order ${m.order_id || "not given"}, ${m.filename || "unnamed"}, ${m.submitted_at.slice(0, 10)}`)))) : null));
}

function photosSection(claim, local) {
  const used = new Set();
  const urlFor = (photo) => {
    const i = local.findIndex((item, j) => !used.has(j) && item.file.name === photo.filename);
    if (i < 0) return null;
    used.add(i);
    return local[i].url;
  };
  if (claim.photos.length === 1) return photoPanel(claim.photos[0], 0, urlFor(claim.photos[0]), false);
  const panels = claim.photos.map((p, i) => photoPanel(p, i, urlFor(p)));
  const tabs = claim.photos.map((p, i) => el("button", {
    type: "button", class: "tab", role: "tab", "aria-selected": i === 0 ? "true" : "false", "aria-controls": `photo-${i}`,
    style: `--level:${LEVELS[p.risk.level].colour}`,
    onclick: () => {
      tabs.forEach((t, j) => t.setAttribute("aria-selected", j === i ? "true" : "false"));
      panels.forEach((panel, j) => { panel.hidden = j !== i; });
    },
  }, el("span", { class: "dot" }), `Photo ${i + 1}`, el("span", { class: "hint" }, `score ${p.risk.score}`)));
  panels.forEach((panel, j) => { panel.hidden = j !== 0; });
  return el("div", { class: "photo-panel" }, el("div", { class: "tabs", role: "tablist", "aria-label": "Photos" }, tabs), panels);
}

function renderClaim(claim, local) {
  results.replaceChildren(...[
    verdict(claim),
    whySection(claim),
    acrossSection(claim),
    rejectedSection(claim),
    limitationsSection(claim),
    photosSection(claim, local),
    el("details", { class: "raw panel section" }, el("summary", {}, "Full JSON report"),
      el("pre", {}, JSON.stringify({ ...claim, photos: claim.photos.map((p) => ({ ...p, ela_preview: p.ela_preview ? "(image data omitted)" : null })) }, null, 2))),
    el("p", { class: "disclaimer" }, claim.disclaimer),
  ].filter(Boolean));
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  results.firstElementChild.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
}

loadHealth();
