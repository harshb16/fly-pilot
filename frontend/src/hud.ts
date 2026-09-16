import type { ErrorMessage, HelloMessage, PilotControls, StateMessage } from "./protocol";
import type { CameraMode } from "./scene";
import type { ConnectionStatus } from "./simClient";

export interface HudHandlers {
  onReset: () => void;
  onPause: () => void;
  onResume: () => void;
  onCamera: (mode: CameraMode) => void;
  onSlider: (axis: keyof PilotControls, value: number, holding: boolean) => void;
  onController: (name: "manual" | "expert" | "expert_observing" | "fly_control" | "hybrid_guidance") => void;
}

export function mountHud(root: HTMLElement, handlers: HudHandlers): (model: HudModel) => void {
  root.innerHTML = `
    <div class="topbar">
      <div>
        <div class="brand">FlyPilot</div>
        <div class="sub" id="milestone-sub">Milestone 5 · JSBSim Cessna 172 · MaleCNS + trained decoder</div>
      </div>
      <div class="controller-toggle">
        <button id="btn-manual" type="button" class="active">MANUAL</button>
        <button id="btn-expert" type="button">EXPERT</button>
        <button id="btn-observing" type="button">EXPERT + FLY OBSERVING</button>
        <button id="btn-fly" type="button">FLY CONTROL</button>
        <button id="btn-hybrid" type="button">HYBRID FLY GUIDANCE</button>
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
    <div class="panel error" id="error-panel" role="alert" hidden></div>
    <div class="panel expert" id="expert-panel" hidden>
      <div class="expert-title">Conventional autopilot — not MaleCNS</div>
      <div class="readout"><span>PHASE</span><strong id="ex-phase">—</strong></div>
      <div class="readout"><span>TGT IAS</span><strong id="ex-ias">—</strong><em>kt</em></div>
      <div class="readout"><span>GS ERR</span><strong id="ex-gs">—</strong><em>m</em></div>
      <div class="readout"><span>XTK</span><strong id="ex-xtk">—</strong><em>m</em></div>
    </div>
    <div class="panel observing" id="observing-panel" hidden>
      <div class="observing-banner">FLY OBSERVING — NOT CONTROLLING</div>
      <p class="observing-note">ExpertLandingController still flies. MaleCNS watches the rendered approach.</p>
      <div class="eyes">
        <figure>
          <figcaption>Left eye</figcaption>
          <canvas id="fly-left" width="48" height="32" aria-label="Left fly-eye preview"></canvas>
        </figure>
        <figure>
          <figcaption>Right eye</figcaption>
          <canvas id="fly-right" width="48" height="32" aria-label="Right fly-eye preview"></canvas>
        </figure>
      </div>
      <div class="readout"><span>RETINA L</span><strong id="ret-l">—</strong></div>
      <div class="readout"><span>RETINA R</span><strong id="ret-r">—</strong></div>
      <div class="readout"><span>ΔL</span><strong id="ret-dl">—</strong></div>
      <div class="readout"><span>SPIKES/S</span><strong id="spk">—</strong></div>
      <div class="readout"><span>DN MEAN</span><strong id="dn-mean">—</strong><em>Hz</em></div>
      <div class="readout"><span>DN L/R</span><strong id="dn-lr">—</strong></div>
      <div class="readout"><span>R1–R6</span><strong id="n-r1">—</strong></div>
      <div class="readout"><span>DN N</span><strong id="n-dn">—</strong></div>
    </div>
    <div class="panel fly-control" id="fly-panel" hidden>
      <div class="fly-banner">FLY CONTROL</div>
      <p class="observing-note">FIXED MALECNS + TRAINED TEMPORAL DECODER — not biological learning. ExpertLandingController is not in this path.</p>
      <div class="eyes">
        <figure>
          <figcaption>Left eye</figcaption>
          <canvas id="fly-left-ctl" width="48" height="32" aria-label="Left fly-eye preview"></canvas>
        </figure>
        <figure>
          <figcaption>Right eye</figcaption>
          <canvas id="fly-right-ctl" width="48" height="32" aria-label="Right fly-eye preview"></canvas>
        </figure>
      </div>
      <div class="readout"><span>AIL</span><strong id="fc-ail">—</strong></div>
      <div class="readout"><span>ELV</span><strong id="fc-elv">—</strong></div>
      <div class="readout"><span>RDR</span><strong id="fc-rdr">—</strong></div>
      <div class="readout"><span>THR</span><strong id="fc-thr">—</strong></div>
      <div class="readout"><span>SPIKES/S</span><strong id="fc-spk">—</strong></div>
      <div class="readout"><span>DN MEAN</span><strong id="fc-dn">—</strong><em>Hz</em></div>
      <div class="readout"><span>GRU ‖h‖</span><strong id="fc-gru">—</strong></div>
      <div class="readout"><span>DN N</span><strong id="fc-ndn">—</strong></div>
    </div>
    <div class="panel hybrid-guidance" id="hybrid-panel" hidden>
      <div class="fly-banner">HYBRID FLY GUIDANCE</div>
      <p class="observing-note">SAFETY-BOUNDED CONNECTOME-GRAPH BANK RESIDUAL + CONVENTIONAL LATERAL ENVELOPE, GLIDESLOPE, AIRSPEED, FLARE, AND STABILIZATION. Uses aircraft telemetry.</p>
      <div class="readout"><span>ROLL CMD</span><strong id="hg-roll">—</strong><em>°</em></div>
      <div class="readout"><span>PITCH CMD</span><strong id="hg-pitch">—</strong><em>°</em></div>
      <div class="readout"><span>IAS CMD</span><strong id="hg-ias">—</strong><em>kt</em></div>
      <div class="readout"><span>THR TRIM</span><strong id="hg-trim">—</strong></div>
      <div class="readout"><span>AIL</span><strong id="hg-ail">—</strong></div>
      <div class="readout"><span>ELV</span><strong id="hg-elv">—</strong></div>
      <div class="readout"><span>RDR</span><strong id="hg-rdr">—</strong></div>
      <div class="readout"><span>THR</span><strong id="hg-thr">—</strong></div>
      <div class="readout"><span>PHASE</span><strong id="hg-phase">—</strong></div>
      <div class="readout"><span>GRAPH RAW</span><strong id="hg-raw">—</strong><em>°</em></div>
      <div class="readout"><span>RESIDUAL</span><strong id="hg-residual">—</strong><em>°</em></div>
    </div>
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
        Aircraft motion comes from JSBSim. MaleCNS does not write inceptors.
      </p>
    </div>
  `;

  root.querySelector("#btn-reset")?.addEventListener("click", handlers.onReset);
  root.querySelector("#btn-pause")?.addEventListener("click", handlers.onPause);
  root.querySelector("#btn-resume")?.addEventListener("click", handlers.onResume);
  root.querySelector("#btn-chase")?.addEventListener("click", () => handlers.onCamera("chase"));
  root.querySelector("#btn-cockpit")?.addEventListener("click", () => handlers.onCamera("cockpit"));
  root.querySelector("#btn-manual")?.addEventListener("click", () => handlers.onController("manual"));
  root.querySelector("#btn-expert")?.addEventListener("click", () => handlers.onController("expert"));
  root.querySelector("#btn-observing")?.addEventListener("click", () => handlers.onController("expert_observing"));
  root.querySelector("#btn-fly")?.addEventListener("click", () => handlers.onController("fly_control"));
  root.querySelector("#btn-hybrid")?.addEventListener("click", () => handlers.onController("hybrid_guidance"));

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
  error: ErrorMessage | null;
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
    const label = state.paused && state.episode.status === "in_progress"
      ? "READY — CHOOSE A MODE OR RESUME"
      : state.episode.status.replaceAll("_", " ").toUpperCase();
    episode.textContent = state.episode.reason ? `${label} — ${state.episode.reason}` : label;
    episode.className = `panel episode ${state.episode.status}`;
  }

  const errorPanel = root.querySelector<HTMLElement>("#error-panel");
  if (errorPanel) {
    errorPanel.hidden = model.error === null;
    errorPanel.textContent = model.error
      ? `${model.error.message}${model.error.action ? ` ${model.error.action}` : ""}`
      : "";
  }

  const controller = state?.controller ?? model.hello?.controller ?? "manual";
  const flown = controller === "expert" || controller === "expert_observing" || controller === "fly_control" || controller === "hybrid_guidance";
  const shown = flown && state ? state.controls : model.localControls;
  syncSlider(root, "aileron", shown.aileron);
  syncSlider(root, "elevator", shown.elevator);
  syncSlider(root, "rudder", shown.rudder);
  syncSlider(root, "throttle", shown.throttle);

  root.querySelector("#btn-manual")?.classList.toggle("active", controller === "manual");
  root.querySelector("#btn-expert")?.classList.toggle("active", controller === "expert");
  root.querySelector("#btn-observing")?.classList.toggle("active", controller === "expert_observing");
  root.querySelector("#btn-fly")?.classList.toggle("active", controller === "fly_control");
  root.querySelector("#btn-hybrid")?.classList.toggle("active", controller === "hybrid_guidance");
  setModeAvailability(root, model.hello);

  const expertPanel = root.querySelector<HTMLElement>("#expert-panel");
  if (expertPanel) {
    const show = controller === "expert" || controller === "expert_observing";
    expertPanel.hidden = !show;
    if (show && state?.expert) {
      setText(root, "ex-phase", String(state.expert.phase).replaceAll("_", " "));
      setText(root, "ex-ias", fmt(state.expert.target_airspeed_kts, 0));
      setText(root, "ex-gs", fmt(state.expert.glideslope_error_m, 1));
      setText(root, "ex-xtk", fmt(state.expert.centerline_error_m, 1));
    }
  }

  const observingPanel = root.querySelector<HTMLElement>("#observing-panel");
  if (observingPanel) {
    const show = controller === "expert_observing";
    observingPanel.hidden = !show;
    const fly = state?.fly_observing;
    if (show && fly) {
      setText(root, "ret-l", fmt(fly.retina.left_mean_luminance, 2));
      setText(root, "ret-r", fmt(fly.retina.right_mean_luminance, 2));
      setText(root, "ret-dl", fmt(fly.retina.mean_temporal, 3));
      setText(root, "spk", fmt(fly.spikes_per_sec, 0));
      setText(root, "dn-mean", fmt(fly.descending.mean_hz, 2));
      setText(root, "dn-lr", `${fmt(fly.descending.left_mean_hz, 2)}/${fmt(fly.descending.right_mean_hz, 2)}`);
      setText(root, "n-r1", String(fly.n_r1r6));
      setText(root, "n-dn", String(fly.n_descending));
    }
  }

  const flyPanel = root.querySelector<HTMLElement>("#fly-panel");
  if (flyPanel) {
    const show = controller === "fly_control";
    flyPanel.hidden = !show;
    const fc = state?.fly_control;
    const fly = state?.fly_observing;
    if (show && fc) {
      setText(root, "fc-ail", fmt(fc.aileron, 2));
      setText(root, "fc-elv", fmt(fc.elevator, 2));
      setText(root, "fc-rdr", fmt(fc.rudder, 2));
      setText(root, "fc-thr", fmt(fc.throttle, 2));
      setText(root, "fc-gru", fmt(fc.gru_hidden_norm, 2));
    }
    if (show && fly) {
      setText(root, "fc-spk", fmt(fly.spikes_per_sec, 0));
      setText(root, "fc-dn", fmt(fly.descending.mean_hz, 2));
      setText(root, "fc-ndn", String(fly.n_descending));
    }
  }

  const hybridPanel = root.querySelector<HTMLElement>("#hybrid-panel");
  if (hybridPanel) {
    const show = controller === "hybrid_guidance";
    hybridPanel.hidden = !show;
    const hybrid = state?.hybrid_guidance;
    if (show && hybrid) {
      setText(root, "hg-roll", fmt(hybrid.roll_command_deg, 1));
      setText(root, "hg-pitch", fmt(hybrid.pitch_command_deg, 1));
      setText(root, "hg-ias", fmt(hybrid.target_airspeed_kts, 0));
      setText(root, "hg-trim", fmt(hybrid.throttle_trim, 2));
      setText(root, "hg-ail", fmt(hybrid.aileron, 2));
      setText(root, "hg-elv", fmt(hybrid.elevator, 2));
      setText(root, "hg-rdr", fmt(hybrid.rudder, 2));
      setText(root, "hg-thr", fmt(hybrid.throttle, 2));
      setText(root, "hg-phase", hybrid.phase ?? "—");
      setText(root, "hg-raw", fmt(hybrid.graph_roll_command_deg, 1));
      setText(root, "hg-residual", fmt(hybrid.graph_roll_residual_deg, 1));
    }
  }

  const sub = root.querySelector("#milestone-sub");
  if (sub) {
    if (controller === "hybrid_guidance") {
      sub.textContent = "Research platform · HYBRID FLY GUIDANCE · connectome topology + conventional stabilization";
    } else if (controller === "fly_control") {
      sub.textContent = "Milestone 5 · FLY CONTROL · fixed MaleCNS + trained temporal decoder";
    } else if (controller === "expert_observing") {
      sub.textContent = "Milestone 5 · EXPERT + FLY OBSERVING · fly is not controlling";
    } else if (controller === "expert") {
      sub.textContent = "Milestone 5 · JSBSim Cessna 172 · conventional expert autopilot";
    } else {
      sub.textContent = "Milestone 5 · JSBSim Cessna 172 · manual control";
    }
  }

  const integrity = root.querySelector("#integrity");
  if (integrity && model.hello) {
    integrity.textContent = model.hello.integrity.note;
  }
}

function setModeAvailability(root: HTMLElement, hello: HelloMessage | null): void {
  const pairs = [
    ["manual", "btn-manual"],
    ["expert", "btn-expert"],
    ["expert_observing", "btn-observing"],
    ["fly_control", "btn-fly"],
    ["hybrid_guidance", "btn-hybrid"],
  ] as const;
  for (const [mode, id] of pairs) {
    const button = root.querySelector<HTMLButtonElement>(`#${id}`);
    const capability = hello?.capabilities?.[mode];
    if (!button) continue;
    button.disabled = capability?.available === false;
    button.title = capability?.available === false
      ? [capability.reason, capability.action].filter(Boolean).join(" ")
      : "";
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
