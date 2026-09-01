import * as THREE from "three";
import * as GaussianSplats3D from "./vendor/gaussian-splats-3d.module.js";

export class UnifiedSceneViewer {
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
    const initialPosition = center.clone().add(
      new THREE.Vector3(radius * 1.1, -radius * 0.64, -radius * 1.3),
    );

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

    // Draw annotations after the Gaussian pass and clear only the depth buffer.
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
    this.showAnnotation(sceneData.annotation);
  }

  showAnnotation(annotation) {
    this.clearAnnotations();
    if (!annotation?.center || !annotation?.size) return;
    this.addBox(annotation.center, annotation.size, 0x35d2ae, 1.0);
    if (annotation.heading) {
      this.addArrow(annotation.center, annotation.size, annotation.heading, 0x35d2ae);
    }
    this.viewer?.forceRenderNextFrame();
  }

  setAnnotationsVisible(visible) {
    if (this.annotationGroup) this.annotationGroup.visible = visible;
    this.viewer?.forceRenderNextFrame();
  }

  clearAnnotations() {
    if (!this.annotationGroup) return;
    while (this.annotationGroup.children.length) {
      const child = this.annotationGroup.children[0];
      this.annotationGroup.remove(child);
      child.traverse?.((object) => {
        object.geometry?.dispose?.();
        if (Array.isArray(object.material)) {
          object.material.forEach((material) => material.dispose?.());
        } else {
          object.material?.dispose?.();
        }
      });
    }
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
    for (let y = 0; y < 2; y += 1) for (let z = 0; z < 2; z += 1) {
      edges.push([index(0, y, z), index(1, y, z)]);
    }
    for (let x = 0; x < 2; x += 1) for (let z = 0; z < 2; z += 1) {
      edges.push([index(x, 0, z), index(x, 1, z)]);
    }
    for (let x = 0; x < 2; x += 1) for (let y = 0; y < 2; y += 1) {
      edges.push([index(x, y, 0), index(x, y, 1)]);
    }
    const radius = Math.max(
      0.005,
      Math.min(...sizeValues.map((value) => Math.abs(value))) * 0.025,
    );
    edges.forEach(([start, end]) => {
      this.annotationGroup.add(
        this.cylinderBetween(corners[start], corners[end], radius, color, opacity),
      );
    });
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
    const shaft = this.cylinderBetween(
      origin,
      shaftEnd,
      Math.max(0.008, length * 0.025),
      color,
      1,
    );
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
    const position = center.clone().add(
      new THREE.Vector3(radius * 1.1, -radius * 0.64, -radius * 1.3),
    );
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
      // v0.4.7 removes a custom rootElement from document.body during dispose().
      // Reparent it first so repeated scene loads cannot throw removeChild().
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
