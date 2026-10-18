"""Websocket server for protocol-v1 frames.

Runs its own asyncio loop in a daemon thread so the simulation loop never
blocks on the network. Each client always gets the *latest* frame (slow
clients drop frames instead of building a backlog). Plain HTTP requests on
the same port serve a small inspector page and ``/hello.json``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import threading
import time
from pathlib import Path
from typing import Optional

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from threedfly.stream.protocol import decode_message

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
_ROUTES = {"/": "inspector.html", "/index.html": "inspector.html"}


class BrainStreamServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        max_hz: float = 30.0,
        compression: Optional[str] = "deflate",
        static_dir: Path = STATIC_DIR,
    ):
        self.host = host
        self.port = int(port)
        self.max_hz = float(max_hz)
        self.compression = compression
        self.static_dir = Path(static_dir)
        self._hello: Optional[bytes] = None
        self._latest: Optional[bytes] = None
        self._seq = 0
        self._last_publish = 0.0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._new: Optional[asyncio.Event] = None
        self._stop: Optional[asyncio.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self.n_clients = 0
        self.frames_published = 0
        self.bytes_published = 0
        self.frames_sent = 0
        self.bytes_sent = 0
        self.t_start = time.time()

    # ------------------------------------------------------------ lifecycle
    def start(self, timeout: float = 10.0) -> "BrainStreamServer":
        self._thread = threading.Thread(target=self._run, name="brain-stream", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("brain stream server did not start")
        if self._error is not None:
            raise RuntimeError(f"brain stream server failed: {self._error}") from self._error
        return self

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as exc:  # pragma: no cover - surfaced via start()
            self._error = exc
            self._ready.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._new = asyncio.Event()
        self._stop = asyncio.Event()
        async with serve(
            self._handler,
            self.host,
            self.port,
            compression=self.compression,
            process_request=self._process_request,
            max_size=None,
        ) as srv:
            sock = next(iter(srv.sockets))
            self.port = int(sock.getsockname()[1])
            self._ready.set()
            await self._stop.wait()

    def stop(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("0.0.0.0", "", None) else self.host
        return f"ws://{host}:{self.port}"

    # ------------------------------------------------------------ producer
    def set_hello(self, hello: bytes) -> None:
        self._hello = bytes(hello)

    def wants_frame(self, now: Optional[float] = None) -> bool:
        """True when a client is connected and ≥ 1/max_hz since the last frame."""
        if self.n_clients <= 0:
            return False
        now = time.perf_counter() if now is None else now
        return (now - self._last_publish) >= (1.0 / self.max_hz if self.max_hz > 0 else 0.0)

    def publish(self, frame: bytes) -> None:
        self._latest = bytes(frame)
        self._seq += 1
        self._last_publish = time.perf_counter()
        self.frames_published += 1
        self.bytes_published += len(frame)
        if self._loop is not None and self._new is not None:
            self._loop.call_soon_threadsafe(self._pulse)

    def _pulse(self) -> None:
        # set() resolves every current waiter; clear() re-arms for the next frame.
        self._new.set()
        self._new.clear()

    def stats(self) -> dict:
        dt = max(time.time() - self.t_start, 1e-9)
        return {
            "url": self.url,
            "clients": self.n_clients,
            "max_hz": self.max_hz,
            "frames_published": self.frames_published,
            "bytes_published": self.bytes_published,
            "frames_sent": self.frames_sent,
            "bytes_sent": self.bytes_sent,
            "uptime_s": dt,
            "last_frame_bytes": len(self._latest) if self._latest else 0,
        }

    # ------------------------------------------------------------ consumer
    async def _handler(self, ws: ServerConnection) -> None:
        self.n_clients += 1
        try:
            while self._hello is None:
                await asyncio.sleep(0.05)
            await ws.send(self._hello)
            seen = 0
            if self._latest is not None:
                await ws.send(self._latest)
                seen = self._seq
                self.frames_sent += 1
                self.bytes_sent += len(self._latest)
            while True:
                if self._seq == seen:
                    await self._new.wait()
                    continue
                buf, seen = self._latest, self._seq
                await ws.send(buf)
                self.frames_sent += 1
                self.bytes_sent += len(buf)
        except ConnectionClosed:
            pass
        finally:
            self.n_clients -= 1

    def _process_request(self, conn: ServerConnection, request: Request):
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        path = request.path.split("?", 1)[0]
        if path == "/hello.json":
            meta = decode_message(self._hello).meta if self._hello else {}
            return _response(200, json.dumps(meta).encode(), "application/json")
        if path == "/stats.json":
            return _response(200, json.dumps(self.stats()).encode(), "application/json")
        name = _ROUTES.get(path, path.lstrip("/"))
        f = (self.static_dir / name).resolve()
        if self.static_dir.resolve() not in f.parents or not f.is_file():
            return _response(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if f.suffix == ".js":
            ctype = "text/javascript"
        return _response(200, f.read_bytes(), ctype)


def _response(code: int, body: bytes, ctype: str) -> Response:
    reason = {200: "OK", 404: "Not Found"}.get(code, "OK")
    headers = Headers(
        [
            ("Content-Type", ctype),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "no-store"),
            ("Access-Control-Allow-Origin", "*"),
        ]
    )
    return Response(code, reason, headers, body)
