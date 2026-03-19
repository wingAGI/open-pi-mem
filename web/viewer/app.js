const state = {
  report: null,
  activeIndex: 0,
  baseDir: "",
  clipAnimationTimer: null,
};

const els = {
  reportFile: document.getElementById("report-file"),
  loadQueryBtn: document.getElementById("load-query-btn"),
  goal: document.getElementById("goal"),
  modelPath: document.getElementById("model-path"),
  source: document.getElementById("source"),
  plannerHz: document.getElementById("planner-hz"),
  frameList: document.getElementById("frame-list"),
  frameMeta: document.getElementById("frame-meta"),
  clipPreview: document.getElementById("clip-preview"),
  videoMeta: document.getElementById("video-meta"),
  inputFrameStrip: document.getElementById("input-frame-strip"),
  detailGoal: document.getElementById("detail-goal"),
  inputMemory: document.getElementById("input-memory"),
  nextSubtask: document.getElementById("next-subtask"),
  nextMemory: document.getElementById("next-memory"),
  prompt: document.getElementById("prompt"),
  rawOutput: document.getElementById("raw-output"),
  prevBtn: document.getElementById("prev-btn"),
  nextBtn: document.getElementById("next-btn"),
};

function init() {
  els.reportFile.addEventListener("change", handleImportFile);
  els.loadQueryBtn.addEventListener("click", loadFromQuery);
  els.prevBtn.addEventListener("click", () => setActiveIndex(state.activeIndex - 1));
  els.nextBtn.addEventListener("click", () => setActiveIndex(state.activeIndex + 1));
  loadFromQuery();
}

async function handleImportFile(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  const text = await file.text();
  const report = JSON.parse(text);
  const baseUrl = URL.createObjectURL(file);
  applyReport(report, baseUrl);
}

async function loadFromQuery() {
  const url = new URL(window.location.href);
  const reportPath = url.searchParams.get("report");
  if (!reportPath) return;
  const response = await fetch(`/repo/${reportPath}`);
  if (!response.ok) return;
  const report = await response.json();
  applyReport(report, `/repo/${reportPath}`);
}

function applyReport(report, sourceUrl) {
  state.report = report;
  state.activeIndex = 0;
  state.baseDir = sourceUrl.includes("/") ? sourceUrl.slice(0, sourceUrl.lastIndexOf("/")) : sourceUrl;
  renderSummary();
  renderFrameList();
  renderActiveFrame();
}

function renderSummary() {
  const report = state.report;
  if (!report) return;
  els.goal.textContent = report.goal || "-";
  els.modelPath.textContent = report.model_path || "-";
  els.source.textContent = report.source || "-";
  els.plannerHz.textContent = `${report.planner_hz ?? "-"} Hz`;
}

function renderFrameList() {
  const report = state.report;
  if (!report?.records?.length) {
    els.frameList.className = "frame-list empty-state";
    els.frameList.textContent = "没有记录。";
    return;
  }
  els.frameList.className = "frame-list";
  els.frameList.innerHTML = "";
  report.records.forEach((record, index) => {
    const btn = document.createElement("button");
    btn.className = `frame-item${index === state.activeIndex ? " active" : ""}`;
    btn.innerHTML = `
      <div class="frame-item-title">Frame ${String(record.frame_index).padStart(3, "0")}</div>
      <div class="frame-item-meta">t=${record.timestamp_sec.toFixed(2)}s</div>
      <div class="frame-item-meta">${record.next_subtask || "(missing subtask)"}</div>
    `;
    btn.addEventListener("click", () => setActiveIndex(index));
    els.frameList.appendChild(btn);
  });
}

function setActiveIndex(index) {
  const report = state.report;
  if (!report?.records?.length) return;
  state.activeIndex = Math.max(0, Math.min(index, report.records.length - 1));
  renderFrameList();
  renderActiveFrame();
}

function renderActiveFrame() {
  const record = state.report?.records?.[state.activeIndex];
  if (!record) return;
  els.frameMeta.textContent = `Frame ${record.frame_index} | t=${record.timestamp_sec.toFixed(2)}s`;
  renderInputFrameStrip(record);
  renderClipPreview(record);
  els.detailGoal.textContent = record.goal || state.report?.goal || "(missing)";
  els.inputMemory.textContent = record.input_memory || "(none)";
  els.nextSubtask.textContent = record.next_subtask || "(missing)";
  els.nextMemory.textContent = record.next_memory || "(missing)";
  els.prompt.textContent = record.prompt || "";
  els.rawOutput.textContent = record.raw_output || "";
}

function toViewerImagePath(imagePath) {
  if (imagePath.startsWith("/")) {
    return `/repo/${imagePath}`;
  }
  return `${state.baseDir}/${imagePath}`;
}

function toViewerSourcePath(path) {
  if (!path) return "";
  if (path.startsWith("/")) {
    return `/repo/${path}`;
  }
  return `${state.baseDir}/${path}`;
}

function renderClipPreview(record) {
  if (state.clipAnimationTimer) {
    clearInterval(state.clipAnimationTimer);
    state.clipAnimationTimer = null;
  }

  const denseSamplingHz = Number(state.report?.dense_sampling_hz || 0);
  const inputPaths = Array.isArray(record.input_image_paths) && record.input_image_paths.length
    ? record.input_image_paths
    : (record.image_path ? [record.image_path] : []);

  if (!inputPaths.length) {
    els.clipPreview.removeAttribute("src");
    els.videoMeta.textContent = "当前记录没有输入帧动图。";
    return;
  }

  const viewerPaths = inputPaths.map((path) => toViewerImagePath(path));
  let currentIndex = 0;
  els.clipPreview.src = viewerPaths[0];

  const frameIndices = inputPaths
    .map((path) => {
      const match = String(path).match(/frame_(\d+)\.(?:png|jpg|jpeg)$/i);
      return match ? Number(match[1]) : null;
    })
    .filter((value) => Number.isFinite(value));

  let startTime = 0;
  let endTime = Number(record.timestamp_sec || 0);
  if (denseSamplingHz > 0 && frameIndices.length) {
    startTime = frameIndices[0] / denseSamplingHz;
    endTime = frameIndices[frameIndices.length - 1] / denseSamplingHz;
  }

  if (viewerPaths.length > 1) {
    const intervalMs = denseSamplingHz > 0 ? Math.max(120, 1000 / denseSamplingHz) : 300;
    state.clipAnimationTimer = window.setInterval(() => {
      currentIndex = (currentIndex + 1) % viewerPaths.length;
      els.clipPreview.src = viewerPaths[currentIndex];
    }, intervalMs);
  }

  els.videoMeta.textContent = `当前输入帧动图: ${startTime.toFixed(2)}s - ${endTime.toFixed(2)}s，共 ${viewerPaths.length} 帧`;
}

function renderInputFrameStrip(record) {
  const inputPaths = record.input_image_paths || (record.image_path ? [record.image_path] : []);
  if (!inputPaths.length) {
    els.inputFrameStrip.className = "input-frame-strip empty-state";
    els.inputFrameStrip.textContent = "当前记录没有多帧输入。";
    return;
  }
  els.inputFrameStrip.className = "input-frame-strip";
  els.inputFrameStrip.innerHTML = "";
  inputPaths.forEach((path, index) => {
    const item = document.createElement("div");
    item.className = "input-frame-thumb";
    item.innerHTML = `
      <img src="${toViewerImagePath(path)}" alt="input frame ${index}" />
      <div class="input-frame-caption">Input ${index + 1}</div>
    `;
    els.inputFrameStrip.appendChild(item);
  });
}

init();
