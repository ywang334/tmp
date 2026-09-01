import * as THREE from "three";
import * as GaussianSplats3D from "./vendor/gaussian-splats-3d.module.js";

const METRIC_LABELS = {
  match_rate: "Object match",
  visibility: "Visibility",
  cross_view_identity: "Cross-view identity",
  position: "Position",
  size: "Size",
  heading_within_30: "Heading within 30 deg",
};

const METRIC_HEAT = {
  match_rate: [30, 132, 103],
  visibility: [49, 105, 178],
  cross_view_identity: [122, 87, 166],
  position: [211, 119, 51],
  size: [31, 145, 159],
  heading_within_30: [185, 72, 105],
};

const OBJECT_COLORS = [
  "#38c9aa", "#f28c52", "#5f91db", "#d36c9d",
  "#b18a42", "#8c79c7", "#40a7a7", "#d0b346",
];

const state = {
  summary: null,
  metric: "position",
  selectedModel: "gpt_56_sol",
  selectedScene: null,
  selectedTarget: null,
  sceneData: null,
  viewer: null,
};

const el = (id) => document.getElementById(id);
const clamp01 = (value) => Math.max(0, Math.min(1, Number(value) || 0));
const pct = (value, digits = 1) => `${(clamp01(value) * 100).toFixed(digits)}%`;
const prettyLabel = (value) => String(value || "unknown").replaceAll("_", " ");
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
}[character]));

function showFatal(error) {
  const box = el("fatal-error");
  box.textContent = `Visualization failed to load\n${error?.stack || error}`;
  box.classList.remove("is-hidden");
  console.error(error);
}

function hideFatal() {
  el("fatal-error").classList.add("is-hidden");
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

function metricValue(model, metric) {
  return model.metrics?.combined?.[metric] ?? 0;
}

function modelByKey(key) {
  return state.summary.models.find((model) => model.key === key) || null;
}

function queryById(id) {
  return state.sceneData?.queries.find((query) => query.id === id) || null;
}

function queryColor(queryId) {
  const index = Math.max(0, state.sceneData?.queries.findIndex((query) => query.id === queryId) ?? 0);
  return OBJECT_COLORS[index % OBJECT_COLORS.length];
}

function initHeader() {
  const benchmark = state.summary.benchmark;
  el("benchmark-counts").innerHTML = [
    ["Scenes", benchmark.scenes],
    ["Objects", benchmark.queries],
    ["Models", state.summary.models.length],
  ].map(([label, value]) => (
    `<span class="benchmark-count"><span>${label}</span><strong>${Number(value).toLocaleString()}</strong></span>`
  )).join("");
}

function initOverview() {
  const select = el("metric-select");
  select.innerHTML = state.summary.metric_order.map((key) => (
    `<option value="${key}" ${key === state.metric ? "selected" : ""}>${METRIC_LABELS[key]}</option>`
  )).join("");
  select.addEventListener("change", () => {
    state.metric = select.value;
    renderOverview();
  });
  renderOverview();
}

function renderOverview() {
  renderRanking();
  renderProfile();
  renderMatrix();
}

function renderRanking() {
  const models = [...state.summary.models].sort((left, right) => (
    metricValue(right, state.metric) - metricValue(left, state.metric)
  ));
  el("ranking-scope").textContent = METRIC_LABELS[state.metric];
  el("ranking-list").innerHTML = models.map((model, index) => {
    const value = metricValue(model, state.metric);
    return `
      <div class="ranking-row ${model.key === state.selectedModel ? "is-selected" : ""}" data-model-key="${model.key}">
        <button type="button" aria-label="Inspect ${escapeHtml(model.name)}">
          <span class="rank-number">${String(index + 1).padStart(2, "0")}</span>
          <span class="rank-name">${escapeHtml(model.name)}</span>
          <span class="rank-track"><span class="rank-fill" style="width:${pct(value)}"></span></span>
          <span class="rank-value">${pct(value)}</span>
        </button>
      </div>`;
  }).join("");
  el("ranking-list").querySelectorAll("[data-model-key]").forEach((row) => {
    row.addEventListener("click", () => selectModel(row.dataset.modelKey));
  });
}

function renderProfile() {
  const model = modelByKey(state.selectedModel);
  if (!model) return;
  el("profile-kind").textContent = model.kind === "open-weight" ? "Open weight" : "API model";
  const rows = state.summary.metric_order.map((key) => {
    const value = metricValue(model, key);
    return `
      <div class="profile-metric">
        <div class="profile-metric-head"><span>${METRIC_LABELS[key]}</span><strong>${pct(value)}</strong></div>
        <div class="profile-track"><span class="profile-fill" style="width:${pct(value)}"></span></div>
      </div>`;
  }).join("");
  el("model-profile").innerHTML = `
    <div class="profile-name">${escapeHtml(model.name)}</div>
    <div class="profile-kindline">Six final evaluation metrics</div>
    <div class="profile-metrics">${rows}</div>`;
}

function heatColor(metric, value) {
  const base = METRIC_HEAT[metric] || METRIC_HEAT.position;
  const strength = 0.10 + clamp01(value) * 0.68;
  const mixed = base.map((channel) => Math.round(255 - (255 - channel) * strength));
  return `rgb(${mixed[0]},${mixed[1]},${mixed[2]})`;
}

function heatOutline(metric) {
  const base = METRIC_HEAT[metric] || METRIC_HEAT.position;
  return `rgba(${base[0]},${base[1]},${base[2]},0.78)`;
}

function renderMatrix() {
  const keys = state.summary.metric_order;
  const best = Object.fromEntries(keys.map((key) => [
    key,
    Math.max(...state.summary.models.map((model) => metricValue(model, key))),
  ]));
  const header = `<thead><tr><th>Model</th>${keys.map((key) => `<th><i class="metric-key" style="background:${heatOutline(key)}"></i>${METRIC_LABELS[key]}</th>`).join("")}</tr></thead>`;
  const body = state.summary.models.map((model) => {
    const cells = keys.map((key) => {
      const value = metricValue(model, key);
      const isBest = Math.abs(value - best[key]) < 1e-12;
      const style = `background:${heatColor(key, value)};${isBest ? `font-weight:800;outline:2px solid ${heatOutline(key)};outline-offset:-2px` : ""}`;
      return `<td><span class="metric-cell" style="${style}">${pct(value)}</span></td>`;
    }).join("");
    return `<tr data-model-key="${model.key}"><td>${escapeHtml(model.name)}</td>${cells}</tr>`;
  }).join("");
  el("metric-matrix").innerHTML = `${header}<tbody>${body}</tbody>`;
  el("metric-matrix").querySelectorAll("[data-model-key]").forEach((row) => {
    row.addEventListener("click", () => selectModel(row.dataset.modelKey));
  });
}

function selectModel(modelKey) {
  if (!modelByKey(modelKey)) return;
  state.selectedModel = modelKey;
  renderOverview();
  if (state.sceneData) {
    renderModelScenes();
    updateOverlay();
  }
}

function initSceneControls() {
  const select = el("scene-select");
  select.innerHTML = state.summary.scenes.map((scene) => (
    `<option value="${scene.id}">${escapeHtml(scene.title)} - ${escapeHtml(scene.renderer_label)}</option>`
  )).join("");
  select.value = state.selectedScene;
  select.addEventListener("change", async () => {
    select.disabled = true;
    try {
      await loadScene(select.value);
    } catch (error) {
      showFatal(error);
    } finally {
      select.disabled = false;
    }
  });
  el("first-view-camera").addEventListener("click", () => state.viewer?.setAnchorView());
  el("overview-camera").addEventListener("click", () => state.viewer?.setOverview());
}

async function loadScene(sceneId) {
  hideFatal();
  const descriptor = state.summary.scenes.find((scene) => scene.id === sceneId);
  if (!descriptor) throw new Error(`Unknown scene: ${sceneId}`);
  state.selectedScene = sceneId;
  state.selectedTarget = null;
  state.sceneData = null;
  el("scene-select").value = sceneId;
  el("loading-label").textContent = "Loading reconstructed 3D scene";
  el("loading-state").classList.remove("is-hidden");

  const scene = await fetchJson(descriptor.data);
  state.sceneData = scene;
  state.selectedTarget = scene.queries[0]?.id || null;
  renderSceneShell();
  if (!state.viewer) state.viewer = new UnifiedSceneViewer(el("viewer-root"));
  await state.viewer.load(scene);
  updateOverlay();
  el("loading-state").classList.add("is-hidden");
  window.__SHOWCASE_STATE__ = state;
}

function renderSceneShell() {
  const scene = state.sceneData;
  el("scene-name").textContent = scene.title;
  el("scene-note").textContent = scene.note;
  el("renderer-label").textContent = scene.render.renderer_label;
  el("view-list").innerHTML = scene.views.map((view) => `
    <figure class="view-card">
      <img src="${view.image}" alt="${escapeHtml(scene.title)} ${escapeHtml(view.label)}" loading="lazy">
      <span>${escapeHtml(view.label)}</span>
    </figure>`).join("");

  el("object-count").textContent = `${scene.queries.length} targets`;
  el("query-list").innerHTML = scene.queries.map((query) => {
    const gt = query.ground_truth;
    return `
      <button class="query-button ${query.id === state.selectedTarget ? "is-active" : ""}" data-query-id="${query.id}" type="button">
        <i style="background:${queryColor(query.id)}"></i>
        <span><strong>${escapeHtml(prettyLabel(query.label))}</strong><small>${escapeHtml(query.id)} - pos ${formatCode(gt.position_code, ["x", "y", "z"])}</small></span>
      </button>`;
  }).join("");
  el("query-list").querySelectorAll("[data-query-id]").forEach((button) => {
    button.addEventListener("click", () => selectTarget(button.dataset.queryId));
  });
  renderModelScenes();
}

function selectTarget(queryId) {
  if (!queryById(queryId)) return;
  state.selectedTarget = queryId;
  el("query-list").querySelectorAll("[data-query-id]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.queryId === queryId);
  });
  renderModelScenes();
  updateOverlay();
}

function formatCode(value, axes) {
  if (!value) return "[?, ?, ?]";
  return `[${axes.map((axis) => value[axis] ?? "?").join(", ")}]`;
}

function selectedResultSummary(result) {
  if (!result?.matched || !result.object) return "Selected target not matched";
  const position = Math.round((result.metrics?.position?.value || 0) * 3);
  const size = Math.round((result.metrics?.size?.value || 0) * 3);
  const angle = result.metrics?.heading?.angle_degrees;
  return `P ${position}/3 - S ${size}/3 - H ${angle == null ? "N/A" : `${Math.round(angle)} deg`}`;
}

function renderModelScenes() {
  const query = queryById(state.selectedTarget);
  if (!query) return;
  const gt = query.ground_truth;
  el("prediction-target-title").textContent = `${query.id} - ${prettyLabel(query.label)}`;
  el("prediction-target-gt").innerHTML = `
    <span>GT views ${formatViews(gt.observed_in)}</span>
    <span>position ${formatCode(gt.position_code, ["x", "y", "z"])}</span>
    <span>size ${formatCode(gt.size_code, ["w", "h", "d"])}</span>
    <span>heading ${formatCode(gt.heading_code, ["x", "y", "z"])}</span>`;

  el("model-scene-grid").innerHTML = state.summary.models.map((model) => {
    const result = query.predictions[model.key];
    const object = result?.object;
    return `
      <article class="model-scene-card ${model.key === state.selectedModel ? "is-selected" : ""}" data-model-key="${model.key}">
        <button class="model-scene-head" type="button" aria-pressed="${model.key === state.selectedModel}">
          <span><strong>${escapeHtml(model.name)}</strong><small>${result?.matched ? "Matched" : "Not matched"}</small></span>
          <b>${escapeHtml(selectedResultSummary(result))}</b>
        </button>
        <canvas class="model-scene-canvas" data-model-key="${model.key}" aria-label="${escapeHtml(model.name)} predicted scene"></canvas>
        <div class="model-scene-codes">
          <span><b>position</b>${formatCode(object?.position_code, ["x", "y", "z"])}</span>
          <span><b>size</b>${formatCode(object?.size_code, ["w", "h", "d"])}</span>
          <span><b>heading</b>${formatCode(object?.heading_code, ["x", "y", "z"])}</span>
        </div>
      </article>`;
  }).join("");

  el("model-scene-grid").querySelectorAll(".model-scene-card[data-model-key]").forEach((card) => {
    card.querySelector(".model-scene-head").addEventListener("click", () => selectModel(card.dataset.modelKey));
  });
  requestAnimationFrame(drawAllModelScenes);
}

function formatViews(views) {
  return `[${(views || []).join(", ")}]`;
}

function drawAllModelScenes() {
  el("model-scene-grid").querySelectorAll(".model-scene-canvas").forEach((canvas) => {
    drawModelScene(canvas, canvas.dataset.modelKey);
  });
}

function codePosition(object) {
  const value = object?.position_code;
  if (!value || ["x", "y", "z"].some((axis) => value[axis] == null)) return null;
  return { x: Number(value.x), y: Number(value.y), z: Number(value.z) };
}

function codeExtent(object) {
  const size = object?.size_code || {};
  return {
    x: Math.max(0.17, Number(size.w || 1) * 0.18),
    y: Math.max(0.17, Number(size.h || 1) * 0.18),
    z: Math.max(0.17, Number(size.d || 1) * 0.18),
  };
}

function rgba(hex, alpha) {
  const value = Number.parseInt(hex.slice(1), 16);
  return `rgba(${(value >> 16) & 255},${(value >> 8) & 255},${value & 255},${alpha})`;
}

function drawModelScene(canvas, modelKey) {
  const cssWidth = Math.max(220, Math.round(canvas.clientWidth));
  const cssHeight = Math.max(150, Math.round(canvas.clientHeight));
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, cssWidth, cssHeight);
  context.fillStyle = "#121917";
  context.fillRect(0, 0, cssWidth, cssHeight);

  const scale = Math.min(cssWidth / 7.3, cssHeight / 6.1);
  const origin = { x: cssWidth * 0.52, y: cssHeight * 0.70 };
  const project = (point) => ({
    x: origin.x + (point.x - point.z * 0.42) * scale,
    y: origin.y - (point.y + point.z * 0.28) * scale,
  });

  drawPredictionGrid(context, project);

  const predictions = state.sceneData.queries.map((query, index) => ({
    query,
    index,
    result: query.predictions[modelKey],
    object: query.predictions[modelKey]?.object || null,
    position: codePosition(query.predictions[modelKey]?.object),
  })).filter((item) => item.position);
  predictions.sort((left, right) => right.position.z - left.position.z);

  const selected = predictions.find((item) => item.query.id === state.selectedTarget);
  predictions.filter((item) => item !== selected).forEach((item) => {
    drawPredictionObject(context, project, item, false);
  });
  if (selected) drawPredictionObject(context, project, selected, true);

  if (!selected) {
    context.fillStyle = "rgba(255,255,255,0.72)";
    context.font = "600 11px Arial, sans-serif";
    context.textAlign = "center";
    context.fillText(`${state.selectedTarget} not matched`, cssWidth / 2, cssHeight / 2);
  }

  context.fillStyle = "rgba(224,235,231,0.54)";
  context.font = "9px Arial, sans-serif";
  context.textAlign = "left";
  context.fillText("+Y", 9, 15);
  context.textAlign = "right";
  context.fillText("+X / +Z code space", cssWidth - 9, cssHeight - 9);
}

function drawPredictionGrid(context, project) {
  context.save();
  context.lineWidth = 1;
  for (let value = -2; value <= 2; value += 1) {
    const xStart = project({ x: value, y: -2.15, z: -2 });
    const xEnd = project({ x: value, y: -2.15, z: 2 });
    const zStart = project({ x: -2, y: -2.15, z: value });
    const zEnd = project({ x: 2, y: -2.15, z: value });
    context.strokeStyle = value === 0 ? "rgba(165,188,180,0.28)" : "rgba(165,188,180,0.13)";
    context.beginPath();
    context.moveTo(xStart.x, xStart.y);
    context.lineTo(xEnd.x, xEnd.y);
    context.stroke();
    context.beginPath();
    context.moveTo(zStart.x, zStart.y);
    context.lineTo(zEnd.x, zEnd.y);
    context.stroke();
  }
  const verticalStart = project({ x: 0, y: -2.15, z: 0 });
  const verticalEnd = project({ x: 0, y: 2.2, z: 0 });
  context.strokeStyle = "rgba(165,188,180,0.22)";
  context.beginPath();
  context.moveTo(verticalStart.x, verticalStart.y);
  context.lineTo(verticalEnd.x, verticalEnd.y);
  context.stroke();
  context.restore();
}

function boxCorners(center, extent) {
  const points = [];
  for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) {
    points.push({
      x: center.x + x * extent.x,
      y: center.y + y * extent.y,
      z: center.z + z * extent.z,
    });
  }
  return points;
}

function drawPredictionObject(context, project, item, selected) {
  const color = queryColor(item.query.id);
  const corners = boxCorners(item.position, codeExtent(item.object)).map(project);
  const index = (x, y, z) => (x ? 4 : 0) + (y ? 2 : 0) + (z ? 1 : 0);
  const edges = [];
  for (let y = 0; y < 2; y += 1) for (let z = 0; z < 2; z += 1) edges.push([index(0, y, z), index(1, y, z)]);
  for (let x = 0; x < 2; x += 1) for (let z = 0; z < 2; z += 1) edges.push([index(x, 0, z), index(x, 1, z)]);
  for (let x = 0; x < 2; x += 1) for (let y = 0; y < 2; y += 1) edges.push([index(x, y, 0), index(x, y, 1)]);

  context.save();
  context.strokeStyle = selected ? color : rgba(color, 0.30);
  context.fillStyle = selected ? rgba(color, 0.14) : rgba(color, 0.025);
  context.lineWidth = selected ? 2.6 : 1;
  context.setLineDash(Object.values(item.object?.size_code || {}).some((value) => value == null) ? [4, 3] : []);
  if (selected) {
    context.shadowColor = rgba(color, 0.8);
    context.shadowBlur = 8;
  }
  const face = [corners[index(0, 0, 0)], corners[index(1, 0, 0)], corners[index(1, 1, 0)], corners[index(0, 1, 0)]];
  context.beginPath();
  face.forEach((point, pointIndex) => pointIndex ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
  context.closePath();
  context.fill();
  edges.forEach(([start, end]) => {
    context.beginPath();
    context.moveTo(corners[start].x, corners[start].y);
    context.lineTo(corners[end].x, corners[end].y);
    context.stroke();
  });
  context.restore();

  if (selected) {
    drawPredictionHeading(context, project, item, color);
    const labelPoint = project({
      x: item.position.x,
      y: item.position.y + codeExtent(item.object).y,
      z: item.position.z,
    });
    context.fillStyle = "#f5faf8";
    context.font = "700 10px Arial, sans-serif";
    context.textAlign = "center";
    context.fillText(`${item.query.id} ${prettyLabel(item.query.label)}`, labelPoint.x, labelPoint.y - 8);
  }
}

function drawPredictionHeading(context, project, item, color) {
  const heading = item.object?.heading_code;
  if (!heading || (Number(heading.x) === 0 && Number(heading.z) === 0)) return;
  const start = project(item.position);
  const end = project({
    x: item.position.x + Number(heading.x || 0) * 0.72,
    y: item.position.y,
    z: item.position.z + Number(heading.z || 0) * 0.72,
  });
  const angle = Math.atan2(end.y - start.y, end.x - start.x);
  context.save();
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = 2.5;
  context.beginPath();
  context.moveTo(start.x, start.y);
  context.lineTo(end.x, end.y);
  context.stroke();
  context.beginPath();
  context.moveTo(end.x, end.y);
  context.lineTo(end.x - Math.cos(angle - 0.55) * 8, end.y - Math.sin(angle - 0.55) * 8);
  context.lineTo(end.x - Math.cos(angle + 0.55) * 8, end.y - Math.sin(angle + 0.55) * 8);
  context.closePath();
  context.fill();
  context.restore();
}

function updateOverlay() {
  const query = queryById(state.selectedTarget);
  const model = modelByKey(state.selectedModel);
  if (!query || !model) return;
  const result = query.predictions[state.selectedModel];
  el("overlay-model-name").textContent = model.name;
  state.viewer?.highlight(query, result?.proxy_geometry || null);
}

class UnifiedSceneViewer {
  constructor(container) {
    this.container = container;
    this.viewer = null;
    this.threeScene = null;
    this.overlayScene = null;
    this.annotationGroup = null;
    this.sceneData = null;
    this.host = null;
  }

  async load(sceneData) {
    await this.dispose();
    this.sceneData = sceneData;
    this.host = document.createElement("div");
    this.host.className = "render-host";
    this.container.prepend(this.host);
    this.threeScene = new THREE.Scene();
    this.overlayScene = new THREE.Scene();
    this.annotationGroup = new THREE.Group();
    this.overlayScene.add(this.annotationGroup);

    const bounds = sceneData.render.bounds;
    const center = new THREE.Vector3(...bounds.center);
    const radius = Math.max(0.35, Number(bounds.radius));
    const initialPosition = center.clone().add(new THREE.Vector3(radius * 1.1, -radius * 0.64, -radius * 1.3));
    this.viewer = new GaussianSplats3D.Viewer({
      rootElement: this.host,
      threeScene: this.threeScene,
      cameraUp: sceneData.render.camera_up,
      initialCameraPosition: initialPosition.toArray(),
      initialCameraLookAt: center.toArray(),
      sharedMemoryForWorkers: false,
      gpuAcceleratedSort: false,
      integerBasedSort: false,
      renderMode: GaussianSplats3D.RenderMode.OnChange,
      sphericalHarmonicsDegree: sceneData.render.spherical_harmonics_degree || 0,
      sceneRevealMode: GaussianSplats3D.SceneRevealMode.Instant,
      antialiased: false,
      dynamicScene: false,
      ignoreDevicePixelRatio: false,
    });

    const renderScene = this.viewer.render.bind(this.viewer);
    this.viewer.render = () => {
      renderScene();
      if (!this.viewer?.initialized || !this.overlayScene || !this.viewer.camera) return;
      const renderer = this.viewer.renderer;
      const autoClear = renderer.autoClear;
      renderer.autoClear = false;
      renderer.clearDepth();
      renderer.render(this.overlayScene, this.viewer.camera);
      renderer.autoClear = autoClear;
    };

    const transform = sceneData.render.transform;
    await this.viewer.addSplatScene(sceneData.render.asset, {
      showLoadingUI: false,
      splatAlphaRemovalThreshold: 1,
      progressiveLoad: false,
      position: transform.position,
      rotation: transform.rotation,
      scale: transform.scale,
    });
    this.viewer.splatMesh.setSplatScale(sceneData.render.splat_scale || 1.0);
    this.viewer.start();
    this.setAnchorView();
  }

  clearAnnotations() {
    if (!this.annotationGroup) return;
    while (this.annotationGroup.children.length) {
      const child = this.annotationGroup.children[0];
      this.annotationGroup.remove(child);
      child.traverse?.((object) => {
        object.geometry?.dispose?.();
        if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose?.());
        else object.material?.dispose?.();
      });
    }
  }

  highlight(query, predictionGeometry) {
    if (!query || !this.annotationGroup) return;
    this.clearAnnotations();
    if (query.geometry?.center && query.geometry?.size) {
      this.addBox(query.geometry.center, query.geometry.size, 0x35d2ae, 1.0);
      if (query.geometry.heading_available && query.geometry.heading) {
        this.addArrow(query.geometry.center, query.geometry.size, query.geometry.heading, 0x35d2ae);
      }
    }
    if (predictionGeometry?.center && predictionGeometry?.size) {
      this.addBox(predictionGeometry.center, predictionGeometry.size, 0xff9b55, 0.92);
      if (predictionGeometry.heading) {
        this.addArrow(predictionGeometry.center, predictionGeometry.size, predictionGeometry.heading, 0xff9b55);
      }
    }
    this.viewer?.forceRenderNextFrame();
  }

  addBox(centerValues, sizeValues, color, opacity) {
    const center = new THREE.Vector3(...centerValues);
    const half = new THREE.Vector3(...sizeValues).multiplyScalar(0.5);
    const corners = [];
    for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) {
      corners.push(center.clone().add(new THREE.Vector3(x * half.x, y * half.y, z * half.z)));
    }
    const index = (x, y, z) => ((x ? 4 : 0) + (y ? 2 : 0) + (z ? 1 : 0));
    const edges = [];
    for (let y = 0; y < 2; y += 1) for (let z = 0; z < 2; z += 1) edges.push([index(0, y, z), index(1, y, z)]);
    for (let x = 0; x < 2; x += 1) for (let z = 0; z < 2; z += 1) edges.push([index(x, 0, z), index(x, 1, z)]);
    for (let x = 0; x < 2; x += 1) for (let y = 0; y < 2; y += 1) edges.push([index(x, y, 0), index(x, y, 1)]);
    const radius = Math.max(0.005, Math.min(...sizeValues.map((value) => Math.abs(value))) * 0.025);
    edges.forEach(([start, end]) => this.annotationGroup.add(this.cylinderBetween(corners[start], corners[end], radius, color, opacity)));
  }

  cylinderBetween(start, end, radius, color, opacity = 1) {
    const direction = end.clone().sub(start);
    const length = direction.length();
    const geometry = new THREE.CylinderGeometry(radius, radius, length, 8, 1, false);
    const material = new THREE.MeshBasicMaterial({
      color,
      transparent: opacity < 1,
      opacity,
      depthTest: false,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(start).add(end).multiplyScalar(0.5);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
    mesh.renderOrder = 4;
    return mesh;
  }

  addArrow(centerValues, sizeValues, directionValues, color) {
    const center = new THREE.Vector3(...centerValues);
    const size = new THREE.Vector3(...sizeValues);
    const direction = new THREE.Vector3(...directionValues).normalize();
    const length = Math.max(0.16, Math.min(1.0, Math.max(size.x, size.z) * 0.78));
    const origin = center.clone().add(new THREE.Vector3(0, -size.y * 0.56, 0));
    const shaftEnd = origin.clone().addScaledVector(direction, length * 0.72);
    const shaft = this.cylinderBetween(origin, shaftEnd, Math.max(0.008, length * 0.025), color, 1);
    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(Math.max(0.025, length * 0.09), length * 0.28, 10),
      new THREE.MeshBasicMaterial({ color, depthTest: false, depthWrite: false }),
    );
    cone.position.copy(origin).addScaledVector(direction, length * 0.86);
    cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    cone.renderOrder = 5;
    this.annotationGroup.add(shaft, cone);
  }

  setAnchorView() {
    if (!this.viewer || !this.sceneData) return;
    const camera = this.sceneData.render.anchor_camera;
    if (!camera) return this.setOverview();
    this.setCamera(
      new THREE.Vector3(...camera.position),
      new THREE.Vector3(...camera.look_at),
      Number(camera.fov_y) || 60,
    );
  }

  setOverview() {
    if (!this.viewer || !this.sceneData) return;
    const bounds = this.sceneData.render.bounds;
    const center = new THREE.Vector3(...bounds.center);
    const radius = Math.max(0.35, Number(bounds.radius));
    const position = center.clone().add(new THREE.Vector3(radius * 1.1, -radius * 0.64, -radius * 1.3));
    this.setCamera(position, center, 65);
  }

  setCamera(position, target, fov) {
    const camera = this.viewer?.camera;
    if (!camera) return;
    camera.position.copy(position);
    camera.up.fromArray(this.sceneData.render.camera_up);
    if (Number.isFinite(fov) && camera.isPerspectiveCamera) camera.fov = fov;
    if (this.viewer.controls) {
      this.viewer.controls.target.copy(target);
      this.viewer.controls.update();
    } else {
      camera.lookAt(target);
    }
    camera.updateProjectionMatrix();
    this.viewer.forceRenderNextFrame();
  }

  async dispose() {
    this.clearAnnotations();
    const viewer = this.viewer;
    const host = this.host;
    this.viewer = null;
    this.host = null;

    if (viewer) {
      viewer.stop();
      if (host) {
        host.style.display = "none";
        if (host.parentElement !== document.body) document.body.appendChild(host);
      }
      try {
        await viewer.dispose();
      } catch (error) {
        if (error?.name !== "NotFoundError") throw error;
      } finally {
        if (host?.isConnected) host.remove();
      }
    }

    this.threeScene = null;
    this.overlayScene = null;
    this.annotationGroup = null;
  }
}

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (state.sceneData) drawAllModelScenes();
  }, 100);
});

async function main() {
  state.summary = await fetchJson("./data/summary.json");
  const params = new URLSearchParams(window.location.search);
  if (state.summary.metric_order.includes(params.get("metric"))) state.metric = params.get("metric");
  if (state.summary.models.some((model) => model.key === params.get("model"))) state.selectedModel = params.get("model");
  state.selectedScene = state.summary.scenes.some((scene) => scene.id === params.get("scene"))
    ? params.get("scene")
    : state.summary.scenes[0].id;
  initHeader();
  initOverview();
  initSceneControls();
  await loadScene(state.selectedScene);
}

main().catch(showFatal);
