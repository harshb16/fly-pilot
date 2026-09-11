import type { PilotControls } from "./protocol";

const STICK_RATE = 2.4;
const STICK_RELEASE = 3.2;
const THROTTLE_RATE = 0.55;

export class InputController {
  controls: PilotControls = { aileron: 0, elevator: 0, rudder: 0, throttle: 0.5 };
  private keys = new Set<string>();
  private sliderHeld = new Set<keyof PilotControls>();

  constructor() {
    window.addEventListener("keydown", (event) => {
      if (event.repeat) return;
      this.keys.add(event.code);
      if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space"].includes(event.code)) {
        event.preventDefault();
      }
    });
    window.addEventListener("keyup", (event) => {
      this.keys.delete(event.code);
    });
    window.addEventListener("blur", () => this.keys.clear());
  }

  holdSlider(axis: keyof PilotControls, value: number): void {
    this.sliderHeld.add(axis);
    this.controls[axis] = value;
  }

  releaseSlider(axis: keyof PilotControls): void {
    this.sliderHeld.delete(axis);
  }

  nudge(axis: keyof PilotControls, delta: number): void {
    this.controls[axis] = clamp(this.controls[axis] + delta, axis === "throttle" ? 0 : -1, 1);
  }

  update(dt: number): PilotControls {
    const left = this.down("KeyA", "ArrowLeft");
    const right = this.down("KeyD", "ArrowRight");
    const noseUp = this.down("KeyS", "ArrowDown");
    const noseDown = this.down("KeyW", "ArrowUp");
    const yawLeft = this.down("KeyQ");
    const yawRight = this.down("KeyE");
    const thrUp = this.down("ShiftLeft", "ShiftRight", "Equal");
    const thrDown = this.down("ControlLeft", "ControlRight", "Minus", "KeyZ");

    if (!this.sliderHeld.has("aileron")) {
      this.controls.aileron = springAxis(this.controls.aileron, axisTarget(right, left), dt);
    }
    if (!this.sliderHeld.has("elevator")) {
      this.controls.elevator = springAxis(this.controls.elevator, axisTarget(noseUp, noseDown), dt);
    }
    if (!this.sliderHeld.has("rudder")) {
      this.controls.rudder = springAxis(this.controls.rudder, axisTarget(yawRight, yawLeft), dt);
    }
    if (!this.sliderHeld.has("throttle")) {
      if (thrUp) this.controls.throttle = clamp(this.controls.throttle + THROTTLE_RATE * dt, 0, 1);
      if (thrDown) this.controls.throttle = clamp(this.controls.throttle - THROTTLE_RATE * dt, 0, 1);
    }
    return { ...this.controls };
  }

  private down(...codes: string[]): boolean {
    return codes.some((code) => this.keys.has(code));
  }
}

function axisTarget(positive: boolean, negative: boolean): number {
  return (positive ? 1 : 0) - (negative ? 1 : 0);
}

function springAxis(current: number, target: number, dt: number): number {
  const rate = target === 0 ? STICK_RELEASE : STICK_RATE;
  if (current === target) return current;
  const step = Math.sign(target - current) * rate * dt;
  if (Math.abs(step) > Math.abs(target - current)) return target;
  return clamp(current + step, -1, 1);
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}
