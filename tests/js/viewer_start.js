// node tests/js/viewer_start.js <scenario> <web dir> [viewer.js]: viewer.js session start against stub
// browser APIs; prints one JSON line for tests/test_quest_view_mode.py.
//   double: the button tapped twice (or tap + scripts/quest_usb.sh vr) while the first start is pending
//   fail:   the layer setup fails after the session was granted, then a retry
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const [scenario, web, viewerArg] = process.argv.slice(2);
const viewerPath = viewerArg || path.join(web, "viewer.js");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// any GL / canvas call works; `then` is undefined so an awaited result is not taken for a promise
const anything = new Proxy(function () {}, {
  get: (_, key) => (key === "then" ? undefined : anything), apply: () => anything,
});

function element() {
  return { textContent: "", disabled: true, width: 640, height: 480, addEventListener() {}, getContext: () => anything };
}
const elements = { status: element(), enter: element(), preview: element() };
const log = { requested: 0, ended: 0, frameLoops: 0, failRenderState: false };

class Session {
  constructor() { this.handlers = {}; }
  addEventListener(type, f) { this.handlers[type] = f; }
  async updateRenderState() { if (log.failRenderState) throw new Error("layer refused"); }
  async requestReferenceSpace() { return {}; }
  requestAnimationFrame() { log.frameLoops += 1; }
  async end() { log.ended += 1; if (this.handlers.end) this.handlers.end(); }
}

const context = {
  console, setTimeout, Promise, JSON, Math, Number, String, Array, Error, Float32Array, Uint8Array,
  Blob: class {}, location: { host: "test" },
  document: { getElementById: (id) => elements[id], createElement: () => element() },
  WebSocket: class { send() {} },
  XRWebGLLayer: class {},
  createImageBitmap: async () => ({ close() {} }),
  navigator: { xr: {
    isSessionSupported: async () => true,
    requestSession: async () => { log.requested += 1; await sleep(30); return new Session(); },
  } },
};
context.window = context;
vm.createContext(context);
for (const file of [path.join(web, "view_mode.js"), viewerPath]) {
  vm.runInContext(fs.readFileSync(file, "utf8"), context, { filename: file });
}

(async () => {
  const out = { scenario };
  if (scenario === "double") {
    out.results = await Promise.all([context.macqStartVr(), context.macqStartVr()]);
  } else {
    log.failRenderState = true;
    out.first = await context.macqStartVr();
    log.failRenderState = false;
    out.retry = await context.macqStartVr();
  }
  Object.assign(out, log, { active: context.macqView.xrActive, mode: context.macqView.mode });
  console.log(JSON.stringify(out));
})();
