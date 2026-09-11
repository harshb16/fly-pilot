import "./style.css";
import { FlightScene } from "./scene";
import { mountHud } from "./hud";
import { InputController } from "./input";
import { SimClient, websocketUrl } from "./simClient";
import type { PilotControls } from "./protocol";

const canvas = document.querySelector<HTMLCanvasElement>("#scene");
const hudRoot = document.querySelector<HTMLElement>("#hud");
if (!canvas || !hudRoot) throw new Error("missing #scene or #hud");

const scene = new FlightScene(canvas);
const input = new InputController();
const client = new SimClient(websocketUrl());
let lastControls: PilotControls = { ...input.controls };
let lastHud = 0;

const renderHud = mountHud(hudRoot, {
  onReset: () => client.reset(),
  onPause: () => client.pause(),
  onResume: () => client.resume(),
  onCamera: (mode) => {
    scene.mode = mode;
  },
  onSlider: (axis, value, holding) => {
    if (holding) input.holdSlider(axis, value);
    else {
      input.holdSlider(axis, value);
      input.releaseSlider(axis);
    }
  },
});

window.addEventListener("keydown", (event) => {
  if (event.code === "KeyR") client.reset();
  if (event.code === "KeyC") scene.mode = scene.mode === "chase" ? "cockpit" : "chase";
  if (event.code === "KeyP") {
    if (client.state?.paused) client.resume();
    else client.pause();
  }
});

let lastHello: typeof client.hello = null;
client.onChange(() => {
  if (client.hello && client.hello !== lastHello) {
    lastHello = client.hello;
    scene.setRunway(client.hello.runway);
  }
});

client.connect();

let previous = performance.now();
function frame(now: number): void {
  const dt = Math.min(0.05, (now - previous) / 1000);
  previous = now;
  const controls = input.update(dt);
  if (client.status === "open" && changed(controls, lastControls)) {
    client.sendControls(controls);
    lastControls = controls;
  } else if (client.status === "open" && now - lastHud > 80) {
    client.sendControls(controls);
    lastControls = controls;
  }
  if (client.state) scene.applyState(client.state);
  scene.render();
  if (now - lastHud > 80) {
    lastHud = now;
    renderHud({
      status: client.status,
      hello: client.hello,
      state: client.state,
      camera: scene.mode,
      localControls: controls,
    });
  }
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);

function changed(a: PilotControls, b: PilotControls): boolean {
  return (
    Math.abs(a.aileron - b.aileron) > 0.01 ||
    Math.abs(a.elevator - b.elevator) > 0.01 ||
    Math.abs(a.rudder - b.rudder) > 0.01 ||
    Math.abs(a.throttle - b.throttle) > 0.01
  );
}
