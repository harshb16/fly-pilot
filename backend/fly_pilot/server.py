"""WebSocket server: browser controls in, JSBSim state out."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

from fly_pilot.protocol import (
    controls_from_message,
    hello_payload,
    parse_client_message,
    state_payload,
)
from fly_pilot.sandbox import LandingSandbox

LOGGER = logging.getLogger("fly_pilot")


class SimServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self.sandbox = LandingSandbox()
        self.sandbox.paused = True
        self.clients: set[ServerConnection] = set()
        self._push_hello = False

    async def handler(self, websocket: ServerConnection) -> None:
        first_client = not self.clients
        self.clients.add(websocket)
        if first_client:
            # Opening the UI always starts on short final, even if the
            # process has been alive for a while with nobody connected.
            self.sandbox.reset()
        LOGGER.info("client connected (%s)", len(self.clients))
        try:
            await websocket.send(json.dumps(hello_payload(self.sandbox)))
            await websocket.send(json.dumps(state_payload(self.sandbox.snapshot())))
            async for raw in websocket:
                self._handle_message(raw)
        except Exception:
            LOGGER.exception("client handler failed")
        finally:
            self.clients.discard(websocket)
            if not self.clients:
                self.sandbox.paused = True
            LOGGER.info("client disconnected (%s)", len(self.clients))

    def _handle_message(self, raw: str | bytes) -> None:
        try:
            data = parse_client_message(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("bad client message: %s", exc)
            return
        kind = data["type"]
        if kind == "controls":
            self.sandbox.set_manual_controls(controls_from_message(data))
        elif kind == "reset":
            seed = data.get("seed")
            self.sandbox.reset(seed=int(seed) if seed is not None else None)
        elif kind == "pause":
            self.sandbox.paused = True
        elif kind == "resume":
            if self.sandbox.episode.info.status.value == "in_progress":
                self.sandbox.paused = False
        elif kind == "set_controller":
            name = str(data.get("name", "manual"))
            try:
                self.sandbox.set_controller(name)
            except ValueError as exc:
                LOGGER.warning("%s", exc)
                return
            self._push_hello = True
        else:
            LOGGER.debug("ignored client message type %s", kind)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if not self.clients:
            return
        message = json.dumps(payload)
        stale: list[ServerConnection] = []
        for client in list(self.clients):
            try:
                await client.send(message)
            except Exception:
                stale.append(client)
        for client in stale:
            self.clients.discard(client)

    async def sim_loop(self) -> None:
        loop = asyncio.get_running_loop()
        last = loop.time()
        emit_carry = 0.0
        emit_dt = 1.0 / 30.0
        while True:
            try:
                now = loop.time()
                wall_dt = min(now - last, 0.25)
                last = now
                self.sandbox.step_realtime(wall_dt)
                emit_carry += wall_dt
                if self._push_hello and self.clients:
                    self._push_hello = False
                    await self.broadcast(hello_payload(self.sandbox))
                    await self.broadcast(state_payload(self.sandbox.snapshot()))
                    emit_carry = 0.0
                elif emit_carry >= emit_dt:
                    emit_carry = 0.0
                    if self.clients:
                        await self.broadcast(state_payload(self.sandbox.snapshot()))
                await asyncio.sleep(0.004)
            except Exception:
                LOGGER.exception("sim loop error")
                await asyncio.sleep(0.05)

    async def run(self) -> None:
        LOGGER.info("JSBSim Cessna 172 sandbox on ws://%s:%s", self.host, self.port)
        async with serve(self.handler, self.host, self.port):
            await self.sim_loop()


def main() -> None:
    parser = argparse.ArgumentParser(description="FlyPilot JSBSim Cessna 172 sandbox server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    server = SimServer(host=args.host, port=args.port)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
