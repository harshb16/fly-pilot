import * as THREE from "three";

/** Off-screen fly-eye cameras. Separate from the human chase/cockpit cameras.

No HUD is in the Three.js scene, so these views never include telemetry overlays.
The Cessna mesh is hidden during the fly pass so the observer sees the world,
not the debug airplane.
*/
export const FLY_EYE_WIDTH = 48;
export const FLY_EYE_HEIGHT = 32;
export const FLY_EYE_FOV_DEG = 140;
const LEFT_YAW_DEG = -63.25;
const RIGHT_YAW_DEG = 63.25;

export class FlyEyeRig {
  readonly left: THREE.PerspectiveCamera;
  readonly right: THREE.PerspectiveCamera;
  readonly leftTarget: THREE.WebGLRenderTarget;
  readonly rightTarget: THREE.WebGLRenderTarget;
  private readonly leftPixels = new Uint8Array(FLY_EYE_WIDTH * FLY_EYE_HEIGHT * 4);
  private readonly rightPixels = new Uint8Array(FLY_EYE_WIDTH * FLY_EYE_HEIGHT * 4);

  constructor() {
    this.left = new THREE.PerspectiveCamera(FLY_EYE_FOV_DEG, FLY_EYE_WIDTH / FLY_EYE_HEIGHT, 0.4, 40000);
    this.right = this.left.clone();
    this.leftTarget = new THREE.WebGLRenderTarget(FLY_EYE_WIDTH, FLY_EYE_HEIGHT, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.NearestFilter,
    });
    this.rightTarget = this.leftTarget.clone();
  }

  attach(aircraft: THREE.Object3D): void {
    aircraft.updateMatrixWorld();
    placeWideCamera(this.left, aircraft, LEFT_YAW_DEG);
    placeWideCamera(this.right, aircraft, RIGHT_YAW_DEG);
  }

  render(
    renderer: THREE.WebGLRenderer,
    scene: THREE.Scene,
    hide: THREE.Object3D | null,
    leftCanvas: HTMLCanvasElement | null,
    rightCanvas: HTMLCanvasElement | null,
  ): void {
    const previous = hide?.visible ?? true;
    if (hide) hide.visible = false;
    blit(renderer, scene, this.left, this.leftTarget, this.leftPixels, leftCanvas);
    blit(renderer, scene, this.right, this.rightTarget, this.rightPixels, rightCanvas);
    if (hide) hide.visible = previous;
  }
}

function placeWideCamera(camera: THREE.PerspectiveCamera, aircraft: THREE.Object3D, yawDeg: number): void {
  const yaw = THREE.MathUtils.degToRad(yawDeg);
  const localEye = new THREE.Vector3(0, 1.2, -2.4);
  const localLook = localEye.clone().add(new THREE.Vector3(Math.sin(yaw) * 80, 0, -Math.cos(yaw) * 80));
  const origin = localEye.applyMatrix4(aircraft.matrixWorld);
  const look = localLook.applyMatrix4(aircraft.matrixWorld);
  camera.position.copy(origin);
  camera.up.copy(new THREE.Vector3(0, 1, 0).applyQuaternion(aircraft.quaternion));
  camera.lookAt(look);
  camera.updateMatrixWorld();
}

function blit(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.PerspectiveCamera,
  target: THREE.WebGLRenderTarget,
  pixels: Uint8Array,
  canvas: HTMLCanvasElement | null,
): void {
  renderer.setRenderTarget(target);
  renderer.render(scene, camera);
  renderer.readRenderTargetPixels(target, 0, 0, FLY_EYE_WIDTH, FLY_EYE_HEIGHT, pixels);
  renderer.setRenderTarget(null);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const image = ctx.createImageData(FLY_EYE_WIDTH, FLY_EYE_HEIGHT);
  // WebGL is origin-bottom; flip to canvas origin-top.
  for (let y = 0; y < FLY_EYE_HEIGHT; y += 1) {
    const src = (FLY_EYE_HEIGHT - 1 - y) * FLY_EYE_WIDTH * 4;
    const dst = y * FLY_EYE_WIDTH * 4;
    image.data.set(pixels.subarray(src, src + FLY_EYE_WIDTH * 4), dst);
  }
  ctx.putImageData(image, 0, 0);
}
