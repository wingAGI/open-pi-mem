const STORAGE_KEY = "open-pi-mem-annotator-state-v3";

const state = {
  episodes: [],
  activeEpisodeId: null,
  playbackSegmentEnd: null,
};

const els = {
  videoInput: document.getElementById("video-input"),
  importFile: document.getElementById("import-file"),
  videoPlayer: document.getElementById("video-player"),
  episodeList: document.getElementById("episode-list"),
  segmentsList: document.getElementById("segments-list"),
  breakpointsList: document.getElementById("breakpoints-list"),
  activeTitle: document.getElementById("active-title"),
  activeMeta: document.getElementById("active-meta"),
  currentTime: document.getElementById("current-time"),
  durationTime: document.getElementById("duration-time"),
  breakpointCount: document.getElementById("breakpoint-count"),
  episodeId: document.getElementById("episode-id"),
  episodeGoal: document.getElementById("episode-goal"),
  episodeFps: document.getElementById("episode-fps"),
  videoPath: document.getElementById("video-path"),
  episodeNotes: document.getElementById("episode-notes"),
  addBreakpointBtn: document.getElementById("add-breakpoint-btn"),
  normalizeBreakpointsBtn: document.getElementById("normalize-breakpoints-btn"),
  clearBreakpointsBtn: document.getElementById("clear-breakpoints-btn"),
  rebuildSegmentsBtn: document.getElementById("rebuild-segments-btn"),
  sortSegmentsBtn: document.getElementById("autofill-text-btn"),
  saveEpisodeBtn: document.getElementById("save-episode-btn"),
  exportCurrentBtn: document.getElementById("export-current-btn"),
  exportAllBtn: document.getElementById("export-all-btn"),
  loadSampleBtn: document.getElementById("load-sample-btn"),
};

function init() {
  loadState();
  bindEvents();
  if (!state.episodes.length) loadSampleEpisodes();
  render();
}

function bindEvents() {
  els.videoInput.addEventListener("change", handleVideoFiles);
  els.importFile.addEventListener("change", handleImport);
  els.videoPlayer.addEventListener("timeupdate", handleVideoTimeUpdate);
  els.videoPlayer.addEventListener("loadedmetadata", handleLoadedMetadata);
  document.querySelectorAll("[data-seek]").forEach((button) => {
    button.addEventListener("click", () => seekBy(Number(button.dataset.seek || 0)));
  });
  els.addBreakpointBtn.addEventListener("click", addBreakpointAtCurrentTime);
  els.normalizeBreakpointsBtn.addEventListener("click", normalizeBreakpoints);
  els.clearBreakpointsBtn.addEventListener("click", clearBreakpoints);
  els.rebuildSegmentsBtn.addEventListener("click", rebuildSegmentsFromBreakpoints);
  els.sortSegmentsBtn.addEventListener("click", autofillEmptyText);
  els.saveEpisodeBtn.addEventListener("click", saveEpisodeFields);
  els.exportCurrentBtn.addEventListener("click", () => saveToRepository("current"));
  els.exportAllBtn.addEventListener("click", () => saveToRepository("all"));
  els.loadSampleBtn.addEventListener("click", loadSampleEpisodes);
  [els.episodeId, els.episodeGoal, els.episodeFps, els.videoPath, els.episodeNotes].forEach((input) => {
    input.addEventListener("change", saveEpisodeFields);
    input.addEventListener("blur", saveEpisodeFields);
  });
  window.addEventListener("keydown", handleKeydown);
}

function handleVideoFiles(event) {
  const files = Array.from(event.target.files || []);
  if (!files.length) return;
  files.forEach((file, index) => addVideoEpisode(file, index));
  event.target.value = "";
  persist();
  render();
}

function addVideoEpisode(file, index) {
  const idBase = slugify(file.name.replace(/\.[^.]+$/, "")) || `episode-${Date.now()}-${index}`;
  const episodeId = uniqueEpisodeId(idBase);
  state.episodes.push({
    id: episodeId,
    episodeId,
    goal: "",
    fps: 10,
    videoName: file.name,
    videoPath: file.name,
    videoUrl: URL.createObjectURL(file),
    durationSec: 0,
    notes: "",
    metadata: {},
    breakpoints: [],
    segments: [],
  });
  state.activeEpisodeId = episodeId;
}

function uniqueEpisodeId(base) {
  let candidate = base;
  let idx = 1;
  const taken = new Set(state.episodes.map((episode) => episode.episodeId));
  while (taken.has(candidate)) {
    idx += 1;
    candidate = `${base}_${idx}`;
  }
  return candidate;
}

function getActiveEpisode() {
  return state.episodes.find((episode) => episode.id === state.activeEpisodeId) || null;
}

function render() {
  renderEpisodeList();
  renderActiveEpisode();
  renderBreakpoints();
  renderSegments();
}

function renderEpisodeList() {
  const template = document.getElementById("episode-item-template");
  els.episodeList.innerHTML = "";
  if (!state.episodes.length) {
    els.episodeList.innerHTML = '<div class="empty-state">还没有视频。点击“添加视频”开始。</div>';
    return;
  }
  state.episodes.forEach((episode) => {
    const node = template.content.firstElementChild.cloneNode(true);
    node.querySelector(".episode-name").textContent = episode.episodeId || episode.videoName;
    node.querySelector(".episode-sub").textContent = `${episode.segments.length} 段 | ${episode.breakpoints.length} 断点 | ${episode.goal || episode.videoName}`;
    if (episode.id === state.activeEpisodeId) node.classList.add("active");
    node.addEventListener("click", () => {
      state.activeEpisodeId = episode.id;
      render();
      loadEpisodeIntoPlayer();
    });
    els.episodeList.appendChild(node);
  });
}

function renderActiveEpisode() {
  const episode = getActiveEpisode();
  if (!episode) {
    els.activeTitle.textContent = "未选择视频";
    els.activeMeta.textContent = "先添加一个本地视频。";
    els.videoPlayer.removeAttribute("src");
    els.videoPlayer.load();
    clearEpisodeInputs();
    return;
  }
  els.activeTitle.textContent = episode.episodeId;
  els.activeMeta.textContent = `${episode.videoName} | ${episode.segments.length} 段`;
  els.episodeId.value = episode.episodeId;
  els.episodeGoal.value = episode.goal;
  els.episodeFps.value = episode.fps;
  els.videoPath.value = episode.videoPath;
  els.episodeNotes.value = episode.notes || "";
  els.breakpointCount.textContent = String(episode.breakpoints.length);
  if (episode.videoUrl && els.videoPlayer.src !== episode.videoUrl) {
    els.videoPlayer.src = episode.videoUrl;
    els.videoPlayer.load();
  }
  els.durationTime.textContent = formatTime(episode.durationSec || 0);
}

function clearEpisodeInputs() {
  els.episodeId.value = "";
  els.episodeGoal.value = "";
  els.episodeFps.value = 10;
  els.videoPath.value = "";
  els.episodeNotes.value = "";
  els.currentTime.textContent = "0.00s";
  els.durationTime.textContent = "0.00s";
  els.breakpointCount.textContent = "0";
}

function renderBreakpoints() {
  const episode = getActiveEpisode();
  const template = document.getElementById("breakpoint-item-template");
  els.breakpointsList.innerHTML = "";
  if (!episode || !episode.breakpoints.length) {
    els.breakpointsList.className = "breakpoints-list empty-state";
    els.breakpointsList.textContent = "还没有断点。播放视频后点击“新增断点”。";
    return;
  }
  els.breakpointsList.className = "breakpoints-list";
  episode.breakpoints.forEach((time, index) => {
    const node = template.content.firstElementChild.cloneNode(true);
    node.querySelector(".breakpoint-time").textContent = formatTime(time);
    node.querySelector(".breakpoint-meta").textContent = `断点 ${index + 1}`;
    node.querySelector('[data-action="jump"]').addEventListener("click", () => {
      playInterval(time, episode.breakpoints[index + 1] ?? episode.durationSec ?? time);
    });
    node.querySelector('[data-action="delete"]').addEventListener("click", () => {
      episode.breakpoints = episode.breakpoints.filter((point) => !nearlyEqual(point, time));
      rebuildSegmentsFromBreakpoints(false);
    });
    els.breakpointsList.appendChild(node);
  });
}

function renderSegments() {
  const episode = getActiveEpisode();
  const template = document.getElementById("segment-item-template");
  els.segmentsList.innerHTML = "";
  if (!episode || !episode.segments.length) {
    els.segmentsList.className = "segments-list empty-state";
    els.segmentsList.textContent = "先打断点，系统才会生成片段。";
    return;
  }
  els.segmentsList.className = "segments-list";
  episode.segments.forEach((segment, index) => {
    const node = template.content.firstElementChild.cloneNode(true);
    node.querySelector(".segment-title").textContent = `片段 ${index + 1}`;
    node.querySelector(".segment-meta").innerHTML = `${formatTime(segment.startSec)} - ${formatTime(segment.endSec)} · <span class="status-chip status-${segment.status}">${segment.status}</span>`;
    node.querySelector('[data-action="jump"]').addEventListener("click", () => {
      playInterval(segment.startSec, segment.endSec);
    });
    const textInput = node.querySelector('[data-field="text"]');
    const statusInput = node.querySelector('[data-field="status"]');
    const confInput = node.querySelector('[data-field="confidence"]');
    const notesInput = node.querySelector('[data-field="notes"]');
    textInput.value = segment.text || "";
    statusInput.value = segment.status || "unknown";
    confInput.value = segment.confidence ?? 1;
    notesInput.value = segment.notes || "";
    textInput.addEventListener("input", () => updateSegmentField(segment.id, "text", textInput.value));
    statusInput.addEventListener("change", () => updateSegmentField(segment.id, "status", statusInput.value));
    confInput.addEventListener("input", () => updateSegmentField(segment.id, "confidence", Number(confInput.value || 1)));
    notesInput.addEventListener("input", () => updateSegmentField(segment.id, "notes", notesInput.value));
    els.segmentsList.appendChild(node);
  });
}

function updateSegmentField(segmentId, field, value) {
  const episode = getActiveEpisode();
  if (!episode) return;
  const segment = episode.segments.find((item) => item.id === segmentId);
  if (!segment) return;
  segment[field] = value;
  persist();
}

function handleVideoTimeUpdate() {
  updateTimeDisplay();
  if (state.playbackSegmentEnd != null && (els.videoPlayer.currentTime || 0) >= state.playbackSegmentEnd - 0.01) {
    els.videoPlayer.pause();
    els.videoPlayer.currentTime = state.playbackSegmentEnd;
    state.playbackSegmentEnd = null;
    updateTimeDisplay();
  }
}

function updateTimeDisplay() {
  els.currentTime.textContent = formatTime(els.videoPlayer.currentTime || 0);
}

function handleLoadedMetadata() {
  const episode = getActiveEpisode();
  if (!episode) return;
  episode.durationSec = Number.isFinite(els.videoPlayer.duration) ? els.videoPlayer.duration : 0;
  els.durationTime.textContent = formatTime(episode.durationSec);
  persist();
}

function seekBy(delta) {
  if (!Number.isFinite(els.videoPlayer.duration)) return;
  state.playbackSegmentEnd = null;
  els.videoPlayer.currentTime = clamp((els.videoPlayer.currentTime || 0) + delta, 0, els.videoPlayer.duration);
}

function playInterval(startSec, endSec) {
  state.playbackSegmentEnd = endSec > startSec ? endSec : null;
  els.videoPlayer.currentTime = startSec;
  updateTimeDisplay();
  els.videoPlayer.play().catch(() => {});
}

function addBreakpointAtCurrentTime() {
  const episode = getActiveEpisode();
  if (!episode) {
    alert("先选择一个视频。");
    return;
  }
  const current = roundTime(els.videoPlayer.currentTime || 0);
  if (current <= 0 || (episode.durationSec && current >= episode.durationSec)) {
    alert("断点应在视频内部，不要打在 0 或视频末尾。")
    return;
  }
  if (!episode.breakpoints.some((point) => nearlyEqual(point, current))) {
    episode.breakpoints.push(current);
  }
  rebuildSegmentsFromBreakpoints(false);
}

function normalizeBreakpoints() {
  const episode = getActiveEpisode();
  if (!episode) return;
  episode.breakpoints = normalizePoints(episode.breakpoints, episode.durationSec);
  rebuildSegmentsFromBreakpoints(false);
}

function clearBreakpoints() {
  const episode = getActiveEpisode();
  if (!episode) return;
  episode.breakpoints = [];
  episode.segments = episode.durationSec > 0 ? [makeSegment(0, roundTime(episode.durationSec))] : [];
  persist();
  render();
}

function rebuildSegmentsFromBreakpoints(showAlert = true) {
  const episode = getActiveEpisode();
  if (!episode) return;
  const oldSegments = new Map(episode.segments.map((segment) => [segment.startSec, segment]));
  const points = normalizePoints(episode.breakpoints, episode.durationSec);
  episode.breakpoints = points;
  const boundaries = [0, ...points, roundTime(episode.durationSec || 0)].filter((value, index, arr) => index === 0 || !nearlyEqual(value, arr[index - 1]));
  const nextSegments = [];
  for (let i = 0; i < boundaries.length - 1; i += 1) {
    const startSec = roundTime(boundaries[i]);
    const endSec = roundTime(boundaries[i + 1]);
    const prev = oldSegments.get(startSec);
    nextSegments.push({
      id: prev?.id || cryptoRandomId(),
      startSec,
      endSec,
      text: prev?.text || "",
      status: prev?.status || "success",
      confidence: prev?.confidence ?? 1,
      notes: prev?.notes || "",
    });
  }
  episode.segments = nextSegments;
  persist();
  render();
  if (showAlert) alert(`已根据 ${points.length} 个断点生成 ${nextSegments.length} 个片段。`);
}

function makeSegment(startSec, endSec) {
  return { id: cryptoRandomId(), startSec, endSec, text: "", status: "success", confidence: 1, notes: "" };
}

function autofillEmptyText() {
  const episode = getActiveEpisode();
  if (!episode) return;
  episode.segments.forEach((segment, index) => {
    if (!segment.text.trim()) segment.text = `subtask_${String(index + 1).padStart(2, "0")}`;
  });
  persist();
  render();
}

function saveEpisodeFields() {
  const episode = getActiveEpisode();
  if (!episode) return;
  episode.episodeId = els.episodeId.value.trim() || episode.episodeId;
  episode.goal = els.episodeGoal.value.trim();
  episode.fps = Number(els.episodeFps.value || 10);
  episode.videoPath = els.videoPath.value.trim() || episode.videoName;
  episode.notes = els.episodeNotes.value.trim();
  persist();
  renderEpisodeList();
  renderActiveEpisode();
}

async function saveToRepository(mode) {
  const episodes = mode === "current" ? [getActiveEpisode()].filter(Boolean) : state.episodes;
  if (!episodes.length) {
    alert("没有可保存的 episode。");
    return;
  }
  const invalid = episodes.find((episode) => !episode.goal || !episode.goal.trim());
  if (invalid) {
    alert(`保存前必须填写 Goal。缺失 episode: ${invalid.episodeId}`);
    return;
  }
  const exported = episodes.map((episode) => toExportEpisode(episode));
  const response = await fetch("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ episodes: exported }),
  });
  const result = await response.json();
  if (!response.ok || !result.ok) {
    alert(`保存失败：${result.error || response.statusText}`);
    return;
  }
  persist();
  alert(`已写入仓库\nJSONL: ${result.jsonl_path}\n目录: ${result.episodes_dir}`);
}

function toExportEpisode(episode) {
  const fps = Number(episode.fps || 10);
  const segments = [...episode.segments].sort((a, b) => a.startSec - b.startSec);
  return {
    episode_id: episode.episodeId,
    goal: episode.goal,
    frames: [],
    proprio: [],
    subtasks: segments.map((segment) => ({
      text: segment.text,
      status: segment.status,
      start_index: Math.round(segment.startSec * fps),
      end_index: Math.round(segment.endSec * fps),
      start_time_sec: segment.startSec,
      end_time_sec: segment.endSec,
      confidence: segment.confidence ?? 1,
      notes: segment.notes || "",
    })),
    metadata: {
      video_path: episode.videoPath,
      video_name: episode.videoName,
      duration_sec: roundTime(episode.durationSec || 0),
      fps,
      notes: episode.notes || "",
      source: "open-pi-mem-annotator",
      breakpoints_sec: [...episode.breakpoints],
    },
  };
}

function handleImport(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    importEpisodesText(String(reader.result || ""));
    event.target.value = "";
  };
  reader.readAsText(file, "utf-8");
}

function importEpisodesText(text) {
  const trimmed = text.trim();
  if (!trimmed) return;
  let rows = [];
  try {
    if (trimmed.startsWith("[")) rows = JSON.parse(trimmed);
    else if (trimmed.startsWith("{") && trimmed.endsWith("}")) rows = [JSON.parse(trimmed)];
    else rows = trimmed.split(/\n+/).map((line) => JSON.parse(line));
  } catch (error) {
    alert(`导入失败：${error.message}`);
    return;
  }
  const imported = rows.map(fromImportedEpisode).filter(Boolean);
  if (!imported.length) {
    alert("导入内容里没有有效 episode。");
    return;
  }
  state.episodes.push(...imported);
  state.activeEpisodeId = imported[0].id;
  persist();
  render();
  loadEpisodeIntoPlayer();
}

function fromImportedEpisode(row) {
  const episodeId = row.episode_id || row.episodeId || uniqueEpisodeId(`episode_${Date.now()}`);
  const metadata = row.metadata || {};
  const segments = Array.isArray(row.subtasks)
    ? row.subtasks.map((segment) => ({
        id: cryptoRandomId(),
        text: segment.text || "",
        status: segment.status || "success",
        startSec: segment.start_time_sec ?? indexToSec(segment.start_index, metadata.fps || 10),
        endSec: segment.end_time_sec ?? indexToSec(segment.end_index, metadata.fps || 10),
        confidence: segment.confidence ?? 1,
        notes: segment.notes || "",
      }))
    : [];
  const breakpoints = Array.isArray(metadata.breakpoints_sec)
    ? metadata.breakpoints_sec.map(Number)
    : segments.slice(0, -1).map((segment) => roundTime(segment.endSec));
  return {
    id: cryptoRandomId(),
    episodeId,
    goal: row.goal || "",
    fps: Number(metadata.fps || 10),
    videoName: metadata.video_name || metadata.video_path || `${episodeId}.mp4`,
    videoPath: metadata.video_path || `${episodeId}.mp4`,
    videoUrl: "",
    durationSec: Number(metadata.duration_sec || (segments.at(-1)?.endSec || 0)),
    notes: metadata.notes || "",
    metadata,
    breakpoints: normalizePoints(breakpoints, Number(metadata.duration_sec || (segments.at(-1)?.endSec || 0))),
    segments: segments.sort((a, b) => a.startSec - b.startSec),
  };
}

function loadSampleEpisodes() {
  state.episodes = [{
    id: cryptoRandomId(),
    episodeId: "sample_mug_001",
    goal: "put the mug into the upper cabinet",
    fps: 10,
    videoName: "sample_mug_episode.mp4",
    videoPath: "videos/sample_mug_episode.mp4",
    videoUrl: "",
    durationSec: 17.2,
    notes: "示例；先打断点，再逐段标注。",
    metadata: { source: "sample" },
    breakpoints: [7.8, 11.4],
    segments: [
      { id: cryptoRandomId(), startSec: 0, endSec: 7.8, text: "reach mug handle", status: "success", confidence: 1, notes: "" },
      { id: cryptoRandomId(), startSec: 7.8, endSec: 11.4, text: "grasp mug", status: "failure", confidence: 0.9, notes: "gripper slipped" },
      { id: cryptoRandomId(), startSec: 11.4, endSec: 17.2, text: "regrasp mug", status: "success", confidence: 1, notes: "" },
    ],
  }];
  state.activeEpisodeId = state.episodes[0].id;
  persist();
  render();
}

function handleKeydown(event) {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
  if (event.key.toLowerCase() === "j") seekBy(-1);
  if (event.key.toLowerCase() === "l") seekBy(1);
  if (event.key.toLowerCase() === "k") {
    if (els.videoPlayer.paused) { state.playbackSegmentEnd = null; els.videoPlayer.play(); } else { els.videoPlayer.pause(); }
  }
  if (event.key === "]") addBreakpointAtCurrentTime();
}

function loadEpisodeIntoPlayer() {
  const episode = getActiveEpisode();
  if (!episode || !episode.videoUrl) {
    els.videoPlayer.removeAttribute("src");
    els.videoPlayer.load();
    return;
  }
  els.videoPlayer.src = episode.videoUrl;
  els.videoPlayer.load();
}

function persist() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({
    ...state,
    episodes: state.episodes.map((episode) => ({ ...episode, videoUrl: "" })),
  }));
}

function loadState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return;
  try {
    const saved = JSON.parse(raw);
    state.episodes = Array.isArray(saved.episodes) ? saved.episodes : [];
    state.activeEpisodeId = saved.activeEpisodeId || state.episodes[0]?.id || null;
  } catch (error) {
    console.warn("Failed to load saved state", error);
  }
}

function normalizePoints(points, durationSec) {
  const duration = roundTime(durationSec || 0);
  return [...new Set((points || []).map((value) => roundTime(Number(value))).filter((value) => Number.isFinite(value) && value > 0 && (duration <= 0 || value < duration)))]
    .sort((a, b) => a - b);
}
function roundTime(value) { return Math.round(Number(value) * 100) / 100; }
function formatTime(value) { return `${roundTime(value).toFixed(2)}s`; }
function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }
function slugify(value) { return value.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, ""); }
function cryptoRandomId() { return window.crypto?.randomUUID ? window.crypto.randomUUID() : `id_${Date.now()}_${Math.random().toString(16).slice(2)}`; }
function indexToSec(index, fps) { return index == null || !fps ? 0 : roundTime(Number(index) / Number(fps)); }
function nearlyEqual(a, b) { return Math.abs(Number(a) - Number(b)) < 0.011; }

init();
