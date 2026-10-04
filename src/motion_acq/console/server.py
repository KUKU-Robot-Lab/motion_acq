"""HTTP side of the console: the page, a state stream (SSE) and the operator actions.

Bound to 127.0.0.1 only. Every /api request must carry a local Host; writes
also need the X-Macq-Console header (a cross-site page cannot set it without a
CORS preflight, which is never answered) and, if present, a local Origin.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from aiohttp import web

from motion_acq.console.core import Console
from motion_acq.console.units import UnitError

WEB_DIR = Path(__file__).resolve().parent / "web"
log = logging.getLogger("motion_acq.console")
STREAM_PERIOD_S = 0.5
MAX_BODY = 1 << 16
APP_ID = "macq-console"


def _local_hosts(port: int) -> set[str]:
    return {f"127.0.0.1:{port}", f"localhost:{port}"}


@web.middleware
async def guard(request: web.Request, handler):
    if request.path.startswith("/api/"):
        port = request.app["port"]
        if request.host not in _local_hosts(port):
            return web.json_response({"ok": False, "error": "Host"}, status=403)
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if request.headers.get("X-Macq-Console") != "1":
                return web.json_response({"ok": False, "error": "header"}, status=403)
            if origin is not None and origin not in {f"http://{h}" for h in _local_hosts(port)}:
                return web.json_response({"ok": False, "error": "Origin"}, status=403)
            if (request.content_length or 0) > MAX_BODY:
                return web.json_response({"ok": False, "error": "body too large"}, status=413)
    return await handler(request)


async def _body(request: web.Request) -> dict:
    if not request.can_read_body:
        return {}
    try:
        data = await request.json()
    except (ValueError, UnicodeDecodeError):
        raise web.HTTPBadRequest(text='{"ok": false, "error": "JSON"}', content_type="application/json")
    if not isinstance(data, dict):
        raise web.HTTPBadRequest(text='{"ok": false, "error": "JSON object"}', content_type="application/json")
    return data


def _reply(result: dict) -> web.Response:
    return web.json_response(result, dumps=lambda o: json.dumps(o, ensure_ascii=False))


async def _call(fn, *args, **kwargs) -> web.Response:
    """Run a console action off the event loop; every error becomes ok=false (and is logged)."""
    try:
        result = await asyncio.to_thread(fn, *args, **kwargs)
    except UnitError as exc:
        result = {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - the window shows it; the server keeps running
        log.exception("console action failed")
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return _reply(result)


def build_app(console: Console, *, port: int, on_quit=None) -> web.Application:
    app = web.Application(middlewares=[guard], client_max_size=MAX_BODY)
    app["port"] = port

    async def index(_request):
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        for name in ("console.css", "console.js"):
            version = int((WEB_DIR / name).stat().st_mtime)
            html = html.replace(f"/static/{name}", f"/static/{name}?v={version}")
        return web.Response(text=html, content_type="text/html", charset="utf-8",
                            headers={"Cache-Control": "no-store"})

    async def ping(_request):
        return _reply({"ok": True, "app": APP_ID, "station": console.station.name, "mode": console.mode})

    async def state(_request):
        return _reply(await asyncio.to_thread(console.snapshot))

    async def stream(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream; charset=utf-8",
                                           "Cache-Control": "no-store"})
        await resp.prepare(request)
        try:
            while True:
                snap = await asyncio.to_thread(console.snapshot)
                data = json.dumps(snap, ensure_ascii=False)
                await resp.write(f"event: state\ndata: {data}\n\n".encode())
                await asyncio.sleep(STREAM_PERIOD_S)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    async def log(request):
        key = request.query.get("key", "")
        text = request.query.get("lines", "300")
        lines = max(10, min(600, int(text))) if text.isdigit() else 300
        child = console.sup.get(key)
        return _reply({"ok": True, "key": key, "lines": console.sup.log_tail(key, lines),
                       "path": str(child.log_path) if child else None})

    async def start(request):
        body = await _body(request)
        opts = body.get("opts") or {}
        if not isinstance(opts, dict):
            return _reply({"ok": False, "error": "opts must be an object"})
        return await _call(console.start, str(body.get("key", "")), opts,
                           confirm=bool(body.get("confirm")), token=str(body.get("token") or ""))

    async def stop(request):
        body = await _body(request)
        return await _call(console.stop, str(body.get("key", "")))

    async def force(request):
        body = await _body(request)
        return await _call(console.force_stop, str(body.get("key", "")), confirm=bool(body.get("confirm")))

    async def key(request):
        body = await _body(request)
        target = str(body.get("key") or console.space_target or "")
        return await _call(console.send_key, target, str(body.get("text", "")))

    async def action(request):
        body = await _body(request)
        return await _call(console.action, str(body.get("name", "")), confirm=bool(body.get("confirm")),
                           token=str(body.get("token") or ""))

    async def stop_all(_request):
        return await _call(console.stop_all)

    async def mode(request):
        body = await _body(request)

        def change():
            console.set_mode(str(body.get("mode", "")))
            return {"ok": True, "mode": console.mode}
        return await _call(change)

    async def settings(request):
        body = await _body(request)
        return await _call(lambda: {"ok": True, "settings": console.update_settings(body)})

    async def space_target(request):
        body = await _body(request)
        return await _call(console.set_space_target, body.get("key"))

    async def quit_console(request):
        body = await _body(request)
        if not body.get("confirm"):
            return _reply({"ok": False, "error": "확인이 필요하다"})
        console.intent("quit")
        if on_quit is not None:
            asyncio.get_running_loop().call_soon(on_quit)
        return _reply({"ok": True})

    app.router.add_get("/", index)
    app.router.add_static("/static", WEB_DIR)
    app.router.add_get("/api/ping", ping)
    app.router.add_get("/api/state", state)
    app.router.add_get("/api/stream", stream)
    app.router.add_get("/api/log", log)
    for path, handler in (("start", start), ("stop", stop), ("force", force), ("key", key), ("action", action),
                          ("stop_all", stop_all), ("mode", mode), ("settings", settings),
                          ("space_target", space_target), ("quit", quit_console)):
        app.router.add_post(f"/api/{path}", handler)
    return app
