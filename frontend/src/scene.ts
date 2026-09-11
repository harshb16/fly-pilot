import * as THREE from "three";
import { applyJsbsimPose, createCessna } from "./aircraftMesh";
import type { RunwayInfo, StateMessage } from "./protocol";
import { createEnvironment } from "./runwayMesh";

export type CameraMode = "chase" | "cockpit";

export class FlightScene {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene: THREE.Scene;
  readonly chaseCamera: THREE.PerspectiveCamera;
  readonly cockpitCamera: THREE.PerspectiveCamera;
  aircraft: THREE.Group;
  mode: CameraMode = "chase";
  private readonly sun: THREE.DirectionalLight;
  private env: THREE.Group | null = null;
  private readonly dummy = new THREE.Object3D();

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xb7d4ea);
    this.scene.fog = new THREE.Fog(0xb7d4ea, 1800, 14000);

    this.chaseCamera = new THREE.PerspectiveCamera(62, window.innerWidth / window.innerHeight, 0.4, 40000);
    this.cockpitCamera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.2, 40000);

    const hemi = new THREE.HemisphereLight(0xe8f3ff, 0x7d8c63, 1.05);
    this.scene.add(hemi);
    this.sun = new THREE.DirectionalLight(0xfff3d6, 1.35);
    this.sun.position.set(400, 700, 180);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.camera.left = -400;
    this.sun.shadow.camera.right = 400;
    this.sun.shadow.camera.top = 400;
    this.sun.shadow.camera.bottom = -400;
    this.scene.add(this.sun);

    this.aircraft = createCessna();
    this.scene.add(this.aircraft);
    this.setRunway({
      name: "FP 09",
      lat_deg: 37,
      lon_deg: -122,
      alt_m: 0,
      heading_deg: 90,
      length_m: 1200,
      width_m: 30,
    });

    window.addEventListener("resize", () => this.resize());
  }

  setRunway(runway: RunwayInfo): void {
    if (this.env) this.scene.remove(this.env);
    this.env = createEnvironment(runway);
    this.scene.add(this.env);
  }

  applyState(state: StateMessage): void {
    const p = state.position;
    const a = state.attitude;
    applyJsbsimPose(this.aircraft, p.east_m, p.north_m, p.up_m, a.heading_deg, a.pitch_deg, a.roll_deg);
    const prop = this.aircraft.getObjectByName("prop");
    if (prop) prop.rotation.z += 0.8 + state.controls.throttle * 2.4;
    this.updateCameras();
  }

  currentCamera(): THREE.PerspectiveCamera {
    return this.mode === "cockpit" ? this.cockpitCamera : this.chaseCamera;
  }

  render(): void {
    this.renderer.render(this.scene, this.currentCamera());
  }

  private updateCameras(): void {
    this.aircraft.updateMatrixWorld();
    const chase = this.dummy;
    chase.position.copy(this.aircraft.position);
    chase.rotation.copy(this.aircraft.rotation);
    const back = new THREE.Vector3(0, 3.4, 14);
    back.applyQuaternion(this.aircraft.quaternion);
    this.chaseCamera.position.lerp(this.aircraft.position.clone().add(back), 0.18);
    const look = this.aircraft.position.clone().add(new THREE.Vector3(0, 1.2, 0));
    this.chaseCamera.lookAt(look);

    const cockpit = new THREE.Vector3(0, 0.55, -1.5);
    cockpit.applyQuaternion(this.aircraft.quaternion);
    this.cockpitCamera.position.copy(this.aircraft.position).add(cockpit);
    const forward = new THREE.Vector3(0, 0.35, -40).applyQuaternion(this.aircraft.quaternion);
    this.cockpitCamera.lookAt(this.aircraft.position.clone().add(forward));
  }

  private resize(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.renderer.setSize(w, h);
    this.chaseCamera.aspect = w / h;
    this.cockpitCamera.aspect = w / h;
    this.chaseCamera.updateProjectionMatrix();
    this.cockpitCamera.updateProjectionMatrix();
  }
}
