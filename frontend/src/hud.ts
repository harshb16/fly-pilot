import type { HelloMessage, PilotControls, StateMessage } from "./protocol";
import type { CameraMode } from "./scene";
import type { ConnectionStatus } from "./simClient";

export interface HudHandlers {
  onReset: () => void;
  onPause: () => void;
  onResume: () => void;
  onCamera: (mode: CameraMode) => void;
  onSlider: (axis: keyof PilotControls, value: number, holding: boolean) => void;
}

export function mountHud(root: HTMLElement, handlers: HudHandlers): (model: HudModel) => void {
  root.innerHTML = `
    <div class="topbar">
      <div>
        <div class="brand">FlyPilot</div>
        <div class="sub">Milestone 1 · JSBSim Cessna 172 · manual control</div>
      </div>
      <div id="status-pill" class="pill">connecting</div>
    </div>
    <div class="panel telemetry">
      <div class="readout"><span>IAS</span><strong id="ias">—</strong><em>kt</em></div>
      <div class="readout"><span>ALT AGL</span><strong id="alt">—</strong><em>m</em></div>
      <div class="readout"><span>VS</span><strong id="vs">—</strong><em>fpm</em></div>
      <div class="readout"><span>HDG</span><strong id="hdg">—</strong><em>°</em></div>
      <div class="readout"><span>PITCH</span><strong id="pitch">—</strong><em>°</em></div>
      <div class="readout"><span>ROLL</span><strong id="roll">—</strong><em>°</em></div>
      <div class="readout"><span>DIST</span><strong id="dist">—</strong><em>m</em></div>
      <div class="readout"><span>XTK</span><strong id="xtk">—</strong><em>m</em></div>
    </div>
    <div class="panel episode" id="episode">IN PROGRESS</div>
    <div class="panel controls">
      ${slider("aileron", "Aileron", -1, 1, 0)}
      ${slider("elevator", "Elevator", -1, 1, 0)}
      ${slider("rudder", "Rudder", -1, 1, 0)}
      ${slider("throttle", "Throttle", 0, 1, 0.5)}
      <div class="buttons">
        <button id="btn-reset" type="button">Reset</button>
        <button id="btn-pause" type="button">Pause</button>
        <button id="btn-resume" type="button">Resume</button>
        <button id="btn-chase" type="button">Chase cam</button>
        <button id="btn-cockpit" type="button">Cockpit cam</button>
      </div>
      <p class="help">
        W/S or ↑↓ pitch · A/D or ←→ roll · Q/E yaw · Shift/Ctrl throttle · R reset · C camera
      </p>
      <p class="integrity" id="integrity">
        Aircraft motion comes from JSBSim. MaleCNS is not in the control loop.
      </p>
    </div>
  `;

  root.querySelector("#btn-reset")?.addEventListener("click", handlers.onReset);
  root.querySelector("#btn-pause")?.addEventListener("click", handlers.onPause);
  root.querySelector("#btn-resume")?.addEventListener("click", handlers.onResume);
  root.querySelector("#btn-chase")?.addEventListener("click", () => handlers.onCamera("chase"));
  root.querySelector("#btn-cockpit")?.addEventListener("click", () => handlers.onCamera("cockpit"));

  for (const axis of ["aileron", "elevator", "rudder", "throttle"] as const) {
    const input = root.querySelector<HTMLInputElement>(`#${axis}`);
    input?.addEventListener("input", () => handlers.onSlider(axis, Number(input.value), true));
    input?.addEventListener("pointerup", () => handlers.onSlider(axis, Number(input.value), false));
    input?.addEventListener("change", () => handlers.onSlider(axis, Number(input.value), false));
  }

  return (model) => renderHud(root, model);
}

export interface HudModel {
  status: ConnectionStatus;
  hello: HelloMessage | null;
  state: StateMessage | null;
  camera: CameraMode;
  localControls: PilotControls;
}

function slider(id: string, label: string, min: number, max: number, value: number): string {
  return `
    <label class="slider">
      <span>${label}</span>
      <input id="${id}" type="range" min="${min}" max="${max}" step="0.01" value="${value}" />
      <em id="${id}-val">${value.toFixed(2)}</em>
    </label>
  `;
}

function renderHud(root: HTMLElement, model: HudModel): void {
  const pill = root.querySelector("#status-pill");
  if (pill) {
    pill.textContent = model.status === "open" ? "JSBSim connected" : model.status;
    pill.className = `pill ${model.status}`;
  }
  const state = model.state;
  setText(root, "ias", fmt(state?.velocity.airspeed_kts, 0));
  setText(root, "alt", fmt(state?.position.alt_agl_m, 0));
  setText(root, "vs", fmt(state?.velocity.vertical_speed_fpm, 0));
  setText(root, "hdg", fmt(state?.attitude.heading_deg, 0));
  setText(root, "pitch", fmt(state?.attitude.pitch_deg, 1));
  setText(root, "roll", fmt(state?.attitude.roll_deg, 1));
  setText(root, "dist", fmt(state ? -state.position.along_m : undefined, 0));
  setText(root, "xtk", fmt(state?.position.right_m, 0));

  const episode = root.querySelector("#episode");
  if (episode && state) {
    const label = state.episode.status.replaceAll("_", " ").toUpperCase();
    episode.textContent = state.episode.reason ? `${label} — ${state.episode.reason}` : label;
    episode.className = `panel episode ${state.episode.status}`;
  }

  syncSlider(root, "aileron", model.localControls.aileron);
  syncSlider(root, "elevator", model.localControls.elevator);
  syncSlider(root, "rudder", model.localControls.rudder);
  syncSlider(root, "throttle", model.localControls.throttle);

  const integrity = root.querySelector("#integrity");
  if (integrity && model.hello) {
    integrity.textContent = model.hello.integrity.note;
  }
}

function syncSlider(root: HTMLElement, id: string, value: number): void {
  const input = root.querySelector<HTMLInputElement>(`#${id}`);
  const label = root.querySelector(`#${id}-val`);
  if (input && document.activeElement !== input) input.value = String(value);
  if (label) label.textContent = value.toFixed(2);
}

function setText(root: HTMLElement, id: string, value: string): void {
  const el = root.querySelector(`#${id}`);
  if (el) el.textContent = value;
}

function fmt(value: number | undefined, digits: number): string {
  if (value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}
