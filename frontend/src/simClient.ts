import type { HelloMessage, PilotControls, ServerMessage, StateMessage } from "./protocol";

export type ConnectionStatus = "connecting" | "open" | "closed";

export class SimClient {
  websocket: WebSocket | null = null;
  status: ConnectionStatus = "connecting";
  hello: HelloMessage | null = null;
  state: StateMessage | null = null;
  lastError: string | null = null;
  private readonly listeners = new Set<() => void>();
  private reconnectTimer = 0;

  constructor(private readonly url: string) {}

  onChange(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  connect(): void {
    this.status = "connecting";
    this.emit();
    const ws = new WebSocket(this.url);
    this.websocket = ws;
    ws.onopen = () => {
      this.status = "open";
      this.lastError = null;
      this.emit();
    };
    ws.onclose = () => {
      this.status = "closed";
      this.emit();
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = window.setTimeout(() => this.connect(), 1000);
    };
    ws.onerror = () => {
      this.lastError = "WebSocket error";
    };
    ws.onmessage = (event) => {
      const message = JSON.parse(String(event.data)) as ServerMessage;
      if (message.type === "hello") {
        this.hello = message;
      } else if (message.type === "state") {
        this.state = message;
      }
      this.emit();
    };
  }

  sendControls(controls: PilotControls): void {
    this.send({ type: "controls", ...controls });
  }

  reset(seed?: number): void {
    if (seed === undefined) this.send({ type: "reset" });
    else this.send({ type: "reset", seed });
  }

  pause(): void {
    this.send({ type: "pause" });
  }

  resume(): void {
    this.send({ type: "resume" });
  }

  setController(name: "manual" | "expert" | "expert_observing"): void {
    this.send({ type: "set_controller", name });
  }

  private send(payload: Record<string, unknown>): void {
    if (this.websocket?.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify(payload));
    }
  }

  private emit(): void {
    for (const listener of this.listeners) listener();
  }
}

export function websocketUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws`;
}
