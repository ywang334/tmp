import { UnifiedSceneViewer } from "./unified_scene_viewer.js";

const root = document.getElementById("viewer-root");
const loading = document.getElementById("loading-state");
const status = document.getElementById("status");
const fatal = document.getElementById("fatal-error");
const reloadButton = document.getElementById("reload-scene");
const annotationButton = document.getElementById("toggle-annotation");

const viewer = new UnifiedSceneViewer(root);
let annotationsVisible = true;

async function loadScene() {
  reloadButton.disabled = true;
  loading.classList.remove("is-hidden");
  fatal.classList.add("is-hidden");
  status.textContent = "Loading scene.ksplat";
  try {
    const response = await fetch("./scene.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`scene.json: HTTP ${response.status}`);
    const scene = await response.json();
    await viewer.load(scene);
    viewer.setAnnotationsVisible(annotationsVisible);
    status.textContent = `${scene.title} | ${scene.render.count.toLocaleString()} splats`;
  } catch (error) {
    fatal.textContent = `Visualization failed\n${error?.stack || error}`;
    fatal.classList.remove("is-hidden");
    status.textContent = "Load failed";
    console.error(error);
  } finally {
    loading.classList.add("is-hidden");
    reloadButton.disabled = false;
  }
}

document.getElementById("anchor-view").addEventListener("click", () => viewer.setAnchorView());
document.getElementById("overview").addEventListener("click", () => viewer.setOverview());
reloadButton.addEventListener("click", loadScene);
annotationButton.addEventListener("click", () => {
  annotationsVisible = !annotationsVisible;
  viewer.setAnnotationsVisible(annotationsVisible);
  annotationButton.setAttribute("aria-pressed", String(annotationsVisible));
  annotationButton.textContent = annotationsVisible ? "Hide bbox" : "Show bbox";
});

window.__3DGS_DEMO__ = { viewer, loadScene };
await loadScene();
