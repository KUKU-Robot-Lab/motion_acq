// motion_acq head view: head camera head-locked in the Quest, poses back to the PC.
// Websocket /ws: binary = latest JPEG of the head camera, text = {"type":"status",...};
// we send {"type":"pose", dt, hmd, left, right} every XR frame (see quest_view/convert.py).
"use strict";

const statusEl = document.getElementById("status");
const enterBtn = document.getElementById("enter");
const preview = document.getElementById("preview");
const previewCtx = preview.getContext("2d");

let ws = null;
let latestImage = null; // ImageBitmap
let imageSeq = 0;
let headStatus = null;
let headAge = null;

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { statusEl.textContent = "서버 연결됨"; };
  ws.onclose = () => { statusEl.textContent = "서버 연결 끊김, 다시 연결 중..."; setTimeout(connect, 1000); };
  ws.onmessage = async (ev) => {
    if (typeof ev.data === "string") {
      const msg = JSON.parse(ev.data);
      if (msg.type === "status") {
        headStatus = msg.head; headAge = msg.head_age_s;
        if (!xrSession) {  // page preview: keep the status visible even without camera frames
          if (!latestImage) { previewCtx.fillStyle = "#000"; previewCtx.fillRect(0, 0, preview.width, preview.height); }
          else previewCtx.drawImage(latestImage, 0, 0, preview.width, preview.height);
          drawStatus(previewCtx, 0, 420, 640, 60);
        }
      }
      return;
    }
    const bitmap = await createImageBitmap(new Blob([ev.data], { type: "image/jpeg" }));
    if (latestImage) latestImage.close();
    latestImage = bitmap;
    imageSeq += 1;
    if (!xrSession) { previewCtx.drawImage(bitmap, 0, 0, preview.width, preview.height); drawStatus(previewCtx, 0, 420, 640, 60); }
  };
}
connect();

// ---- status text (same drawing for the page preview and the headset overlay) ----
function fmt(v) { return (v === null || v === undefined) ? "-" : (v >= 0 ? "+" : "") + v.toFixed(1); }

function statusLines() {
  const h = headStatus;
  if (!h || headAge === null || headAge > 1.0) return ["머리 프로세스 없음 (macq head --udp-target 127.0.0.1:47121)", ""];
  const pan = h.meas_pan_deg === null ? null : h.meas_pan_deg - h.home_pan_deg;
  const tilt = h.meas_tilt_deg === null ? null : h.meas_tilt_deg - h.home_tilt_deg;
  const win = h.window_deg || [0, 0];
  const cam = `카메라 home 대비 pan ${fmt(pan)}° / tilt ${fmt(tilt)}°  (범위 ±${win[0]} / ±${win[1]})`;
  if (h.locked) return ["잠김: 보조자가 Space 를 누르면 지금 보는 방향에서 따라갑니다", cam];
  if (h.state === "hold") return ["멈춤: HMD 추적 끊김", cam];
  if (h.state === "idle") return ["기준 잡는 중...", cam];
  return [`따라가는 중   기준 대비 고개 yaw ${fmt(h.rel_yaw_deg)}°  pitch ${fmt(h.rel_pitch_deg)}°`, cam];
}

function drawStatus(ctx, x, y, w, hgt) {
  const lines = statusLines();
  const locked = headStatus && headStatus.locked;
  ctx.fillStyle = "rgba(15, 23, 42, 0.82)";
  ctx.fillRect(x, y, w, hgt);
  ctx.fillStyle = locked ? "#fbbf24" : "#4ade80";
  ctx.font = `${Math.round(hgt * 0.36)}px "Noto Sans KR", system-ui, sans-serif`;
  ctx.fillText(lines[0], x + hgt * 0.2, y + hgt * 0.42);
  ctx.fillStyle = "#e2e8f0";
  ctx.font = `${Math.round(hgt * 0.3)}px "Noto Sans KR", system-ui, sans-serif`;
  ctx.fillText(lines[1], x + hgt * 0.2, y + hgt * 0.85);
}

// ---- WebXR ----
let xrSession = null;
let refSpace = null;
let gl = null;
let program = null;
let quadBuffer = null;
let videoTex = null;
let overlayTex = null;
let uploadedSeq = -1;
const overlay = document.createElement("canvas");
overlay.width = 1024; overlay.height = 160;
const overlayCtx = overlay.getContext("2d");
let lastTime = null;

if (navigator.xr) {
  navigator.xr.isSessionSupported("immersive-vr").then((ok) => {
    enterBtn.disabled = !ok;
    if (!ok) statusEl.textContent += " (이 브라우저는 VR 을 지원하지 않습니다)";
  });
} else {
  statusEl.textContent += " (WebXR 없음: 미리보기만)";
}

enterBtn.addEventListener("click", async () => {
  xrSession = await navigator.xr.requestSession("immersive-vr", { optionalFeatures: ["local-floor"] });
  xrSession.addEventListener("end", () => { xrSession = null; enterBtn.disabled = false; });
  const canvas = document.createElement("canvas");
  gl = canvas.getContext("webgl", { xrCompatible: true });
  setupGl();
  await xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });
  try { refSpace = await xrSession.requestReferenceSpace("local-floor"); }
  catch (e) { refSpace = await xrSession.requestReferenceSpace("local"); }
  enterBtn.disabled = true;
  xrSession.requestAnimationFrame(onXRFrame);
});

function setupGl() {
  const vs = `attribute vec2 a; uniform mat4 m; varying vec2 uv;
    void main() { uv = vec2(a.x * 0.5 + 0.5, 0.5 - a.y * 0.5); gl_Position = m * vec4(a, 0.0, 1.0); }`;
  const fs = `precision mediump float; uniform sampler2D t; varying vec2 uv;
    void main() { gl_FragColor = texture2D(t, uv); }`;
  const compile = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
  program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(program);
  quadBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  videoTex = makeTexture();
  overlayTex = makeTexture();
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
}

function makeTexture() {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
  return t;
}

// column-major 4x4: projection * translate(x, y, z) * scale(sx, sy)
function quadMatrix(proj, x, y, z, sx, sy) {
  const mdl = [sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, 1, 0, x, y, z, 1];
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0; for (let k = 0; k < 4; k++) s += proj[k * 4 + r] * mdl[c * 4 + k]; out[c * 4 + r] = s;
  }
  return out;
}

function drawQuad(tex, matrix) {
  gl.useProgram(program);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  const loc = gl.getAttribLocation(program, "a");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  gl.uniformMatrix4fv(gl.getUniformLocation(program, "m"), false, matrix);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.uniform1i(gl.getUniformLocation(program, "t"), 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

function poseOf(xrPose) {
  if (!xrPose) return null;
  const p = xrPose.transform.position, o = xrPose.transform.orientation;
  return { p: [p.x, p.y, p.z], q: [o.x, o.y, o.z, o.w] };
}

function onXRFrame(time, frame) {
  const session = frame.session;
  session.requestAnimationFrame(onXRFrame);
  const viewer = frame.getViewerPose(refSpace);

  // poses back to the PC, every frame
  const msg = { type: "pose", dt: lastTime === null ? 0 : (time - lastTime) / 1000, hmd: viewer ? poseOf(viewer) : null,
                left: null, right: null };
  lastTime = time;
  for (const src of session.inputSources) {
    if (!src.gripSpace || (src.handedness !== "left" && src.handedness !== "right")) continue;
    const pose = poseOf(frame.getPose(src.gripSpace, refSpace));
    if (!pose) continue;
    if (src.gamepad) {
      pose.buttons = Array.from(src.gamepad.buttons, (b) => b.pressed);
      pose.axes = Array.from(src.gamepad.axes);
    }
    msg[src.handedness] = pose;
  }
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));

  // head camera head-locked 1.2 m ahead, status strip under it
  if (!viewer) return;
  const layer = session.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
  gl.clearColor(0.02, 0.03, 0.05, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  if (latestImage && uploadedSeq !== imageSeq) {
    gl.bindTexture(gl.TEXTURE_2D, videoTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, latestImage);
    uploadedSeq = imageSeq;
  }
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  drawStatus(overlayCtx, 0, 0, overlay.width, overlay.height);
  gl.bindTexture(gl.TEXTURE_2D, overlayTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, overlay);
  for (const view of viewer.views) {
    const vp = layer.getViewport(view);
    gl.viewport(vp.x, vp.y, vp.width, vp.height);
    drawQuad(videoTex, quadMatrix(view.projectionMatrix, 0, 0.05, -1.2, 0.6, 0.45));
    drawQuad(overlayTex, quadMatrix(view.projectionMatrix, 0, -0.48, -1.2, 0.6, 0.094));
  }
}
