"use strict";
// macq console window: renders /api/stream, sends operator actions to /api/*.

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
let S = null;
let drawerKey = null;
let lastEventT = Date.now() / 1000;
const SIDE_KO = { right: "오른", left: "왼" };
const UNIT_KO = { quest_view: "영상 서버", mock_quest: "가짜 Quest", head: "머리", arm: "팔", record: "녹화",
  hand_right: "오른손 노드", hand_left: "왼손 노드", calib_right: "오른손 보정", calib_left: "왼손 보정" };
const unitName = (k) => UNIT_KO[k] || k.replace(/^task_/, "작업 ");
const esc = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 1) => (typeof v === "number" && isFinite(v) ? v.toFixed(d) : "-");

// ---------------------------------------------------------------- requests
async function post(path, body) {
  let r;
  try {
    r = await fetch(`/api/${path}`, { method: "POST", body: JSON.stringify(body || {}),
      headers: { "Content-Type": "application/json", "X-Macq-Console": "1" } });
  } catch (e) {
    return { ok: false, error: `콘솔 서버 응답 없음 (${e})` };
  }
  try {
    return await r.json();
  } catch {
    return { ok: false, error: `콘솔 서버 응답을 읽지 못했다 (HTTP ${r.status})` };
  }
}

async function act(path, body, btn, okText) {
  if (btn) { btn.classList.add("busy"); btn.disabled = true; }
  try {
    let res = await post(path, body);
    for (let round = 0; res.need_confirm && round < 3; round++) {  // the server re-asks if the command changed
      const yes = await confirmDialog(res.title, res.summary, res.command, true);
      if (!yes) { toast("취소했다"); return res; }
      res = await post(path, { ...body, confirm: true, token: res.token });
    }
    toast(res.ok ? (okText || "보냈다") : (res.error || "실패"), res.ok ? "ok" : "bad");
    return res;
  } finally {
    if (btn) { btn.classList.remove("busy"); btn.disabled = false; }
  }
}

function confirmDialog(title, summary, command, danger) {
  const dlg = $("#confirm");
  $("#confirm-title").textContent = title;
  $("#confirm-summary").textContent = summary || "";
  $("#confirm-command").textContent = command || "";
  $("#confirm-command").hidden = !command;
  dlg.classList.toggle("real", !!danger && S?.mode === "real");
  $("#confirm-ok").textContent = danger && S?.mode === "real" ? "실기 실행" : "실행";
  dlg.returnValue = "cancel";
  dlg.showModal();
  return new Promise((resolve) => dlg.addEventListener("close", () => resolve(dlg.returnValue === "ok"), { once: true }));
}

function toast(text, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = text;
  $("#toasts").append(el);
  setTimeout(() => el.remove(), kind === "bad" ? 9000 : 3500);
}

// ---------------------------------------------------------------- rendering helpers
function unitBadge(u, override) {
  if (!u || (!u.running && u.rc === null)) return ["mute", "정지"];
  if (u.running && u.stopping_s !== null && u.stopping_s !== undefined)
    return ["warn", u.forced ? `강제 종료 중 ${fmt(u.stopping_s, 0)} s` : `정지 중 ${fmt(u.stopping_s, 0)} s (안전 자세)`];
  if (u.running) return override || u.phase || ["live", "실행 중"];
  if (u.rc === 0) return ["mute", "끝남"];
  return ["bad", `끝남 (rc ${u.rc}) ${u.phase && u.phase[0] === "bad" ? u.phase[1] : ""}`];
}

function setBadge(el, [tone, text]) {
  el.className = `badge ${tone}`;
  el.textContent = text;
}

function facts(el, rows) {
  el.innerHTML = rows.map(([k, v, tone]) => `<dt>${esc(k)}</dt><dd class="${tone || ""}">${esc(v)}</dd>`).join("");
}

function tail(el, u, n = 6) {
  const lines = (u?.tail || []).slice(-n);
  if (u?.partial) lines.push(u.partial);
  const text = lines.join("\n");
  if (el.textContent !== text) { el.textContent = text; el.scrollTop = el.scrollHeight; }
}

const running = (k) => !!S?.units?.[k]?.running;
const stopping = (k) => running(k) && S.units[k].stopping_s !== null && S.units[k].stopping_s !== undefined;

// ---------------------------------------------------------------- sections
function renderTop(s) {
  const st = s.station;
  $("#station").textContent = st.name;
  $("#robot").textContent = `${st.robot} · 손 도메인 ${st.ros_domain ?? "-"}`;
  document.body.classList.toggle("is-real", s.mode === "real");
  document.body.classList.toggle("is-fake", s.mode === "fake");
  document.body.classList.toggle("has-hands", st.hands.length > 0);
  $("#card-head").hidden = !st.has_head;
  const busy = s.running.length > 0;
  $$("#top [data-mode]").forEach((b) => {
    b.setAttribute("aria-checked", String(b.dataset.mode === s.mode));
    b.disabled = busy && b.dataset.mode !== s.mode;
    b.title = b.disabled ? "실행 중인 것을 모두 정지한 뒤 바꿀 수 있다" : "";
  });
  document.title = `macq 콘솔 · ${st.name} · ${s.mode === "real" ? "실기" : "FAKE"}`;
}

function pill(label, value, tone) {
  return `<div class="pill ${tone}"><small>${esc(label)}</small><b>${esc(value)}</b></div>`;
}

function renderHealth(s) {
  const p = s.probes || {};
  const out = [];
  const q = p.quest || {};
  const qOk = (q.devices || []).length === 1 && q.devices[0] === "device";
  out.push(pill("Quest USB", qOk ? (q.awake === false ? "연결 · 잠듦" : "연결됨") : (q.text || "확인 중"), qOk ? (q.awake === false ? "warn" : "ok") : "bad"));
  const src = { app: "HandUMI 앱", view: "헤드셋 영상", mock: "가짜 Quest", external: "콘솔 밖 프로그램" }[s.quest_source];
  out.push(pill("자세 입력", src || "없음", s.quest_source === "external" ? "bad" : src ? "ok" : "warn"));
  for (const [side, port] of Object.entries(s.station.can_ports)) {
    const c = (p.can || {})[side] || {};
    const ok = c.up && c.fd;
    out.push(pill(`CAN ${SIDE_KO[side]}팔 ${port}`, ok ? "UP FD" : c.exists === false ? "없음" : c.up ? "FD 아님" : "DOWN", ok ? "ok" : s.mode === "real" ? "bad" : "mute"));
  }
  const holders = Array.isArray(p.can_holders) ? p.can_holders : [];
  out.push(pill("s2r CAN 점유", holders.length ? "점유 중" : "없음", holders.length ? "bad" : "ok"));
  if (s.station.has_head) {
    const h = p.head_port || {};
    out.push(pill("머리 포트", !h.exists ? "없음" : (h.holders || []).length ? `PID ${h.holders.join(",")}` : "비어 있음",
      !h.exists ? (s.mode === "real" ? "bad" : "mute") : (h.holders || []).length ? (running("head") ? "live" : "bad") : "ok"));
  }
  const rt = p.rt || {};
  out.push(pill("RT 한도", rt.rtprio !== undefined ? `rtprio ${rt.rtprio}` : "-", rt.ok ? "ok" : "warn"));
  if (s.station.hands.length) {
    const ros = p.ros || {};
    for (const side of s.station.hands) {
      const h = (ros.hands || {})[side];
      out.push(pill(`손 ${SIDE_KO[side]} EtherCAT`, ros.error ? "ROS 오류" : !h ? "확인 중" : h.driver ? "드라이버 있음" : "없음",
        ros.error ? "bad" : !h ? "mute" : h.driver ? "ok" : "warn"));
    }
    const g = p.gloves || {};
    out.push(pill("SenseCom", g.sensecom_started_at ? "실행 중" : "꺼짐", g.sensecom_started_at ? "ok" : "mute"));
    for (const [side, glove] of Object.entries(g.gloves || {}))
      out.push(pill(`장갑 ${SIDE_KO[side]}`, glove.connected ? "연결" : "미연결", glove.connected ? "ok" : "mute"));
  }
  $("#health").innerHTML = out.join("");
}

function renderAlerts(s) {
  const items = [];
  if (s.closing)
    items.push(`<div class="alert warn"><p><b>콘솔 종료 중</b>: 모든 단위를 안전 자세로 정지하는 중이다. 끝나면 서버가 내려간다(새 시작 불가).</p></div>`);
  for (const e of s.left_behind || [])
    items.push(`<div class="alert"><p>이전 콘솔이 남긴 <b>${esc(unitName(e.key))}</b> (PID ${Number(e.pid)}) 가 아직 돈다${e.stopping ? " (정지 중)" : ""}: <code>${esc((e.argv || []).join(" ").slice(0, 160))}</code></p>
      <button type="button" class="btn danger" data-left="${Number(e.pid)}" ${e.stopping ? "disabled" : ""}>정지 (SIGINT)</button></div>`);
  for (const [key, u] of Object.entries(s.units || {})) {
    if (!u.prompt) continue;
    const yn = /\[y\/n\]/i.test(u.prompt);
    items.push(`<div class="alert ${key.startsWith("calib") ? "warn" : ""}"><p><b>${esc(unitName(key))}</b> 이 입력을 기다린다: <code>${esc(u.prompt)}</code></p>
      ${yn ? `<button type="button" class="btn" data-send="${key}" data-text="y&#10;">y</button><button type="button" class="btn" data-send="${key}" data-text="n&#10;">n</button>`
           : `<button type="button" class="btn danger" data-send="${key}" data-text="&#10;">Enter 보내기</button>`}</div>`);
  }
  const html = items.join("");
  if ($("#alerts").dataset.html !== html) { $("#alerts").innerHTML = html; $("#alerts").dataset.html = html; }
}

function renderQuest(s) {
  const card = $("#card-quest");
  const q = (s.probes || {}).quest || {};
  const src = { app: ["ok", "HandUMI 앱"], view: ["live", "헤드셋 영상"], mock: ["live", "가짜 Quest"],
    external: ["bad", "콘솔 밖 프로그램"] }[s.quest_source] || ["warn", "입력 없음"];
  setBadge($('[data-f="source"]', card), src);
  const qv = s.quest_view;
  const rows = [["USB", q.text || "확인 중", (q.devices || []).length === 1 && q.devices[0] === "device" ? "ok" : "bad"],
    ["헤드셋", q.awake === undefined || q.awake === null ? "-" : q.awake ? "깨어 있음" : "잠듦 (쓰면 깬다)", q.awake ? "ok" : "warn"],
    ["앱 forward 65432", q.forward ? "있음" : "없음", q.forward ? "ok" : ""],
    ["영상 reverse 8787", q.reverse ? "있음" : "없음", q.reverse ? "ok" : ""]];
  const owner = (s.probes || {}).quest_port;
  if (s.quest_source === "external" && owner)
    rows.push(["TCP 65432", `콘솔 밖 ${owner.name} (PID ${owner.pid ?? "?"}) 사용 중: 끄고 다시`, "bad"]);
  const uv = s.units.quest_view;
  rows.push(["영상 서버", running("quest_view") ? (uv.phase ? uv.phase[1] : "실행 중") : "꺼짐", running("quest_view") ? "live" : ""]);
  if (qv) rows.push(["헤드셋 자세", `${qv.poses} 개 (마지막 ${qv.last} 전) · 받는 곳 ${qv.clients}`, qv.last === "never" ? "warn" : "ok"]);
  facts($('[data-f="facts"]', card), rows);
  const busyJob = ["quest_app", "quest_view"].some((j) => (s.jobs[j] || {}).state === "running");
  const users = ["head", "arm", "record"].some(running);
  $$('[data-action="quest:app"], [data-action="quest:view"]', card).forEach((b) => {
    b.disabled = busyJob || users;
    b.title = users ? "머리/팔/녹화를 먼저 정지" : "";
  });
  const mock = $('[data-toggle="mock_quest"]', card);
  mock.textContent = running("mock_quest") ? "가짜 Quest 정지" : "가짜 Quest 시작";
  const job = Object.entries(s.jobs).filter(([k]) => k.startsWith("quest")).sort((a, b) => b[1].at - a[1].at)[0];
  const jobEl = $('[data-f="job"]', card);
  if (job) {
    const [name, j] = job;
    jobEl.className = `job ${j.state === "failed" ? "bad" : j.state === "ok" ? "ok" : ""}`;
    jobEl.textContent = `${name === "quest_app" ? "앱 모드" : "영상 모드"}: ${j.state === "running" ? "진행 중..." : j.text}`;
  }
}

function renderArm(s) {
  const card = $("#card-arm");
  const u = s.units.arm;
  setBadge($('[data-f="phase"]', card), unitBadge(u));
  card.classList.toggle("running", running("arm"));
  card.classList.toggle("stopping", stopping("arm"));
  const locked = running("arm") || running("record");
  $$("[data-side]", card).forEach((b) => {
    b.setAttribute("aria-checked", String(b.dataset.side === s.settings.arm_side));
    b.disabled = locked;
  });
  syncInput("#scale", s.settings.scale);
  $("#scale").disabled = locked;
  const tcp = s.station.tcp;
  $('[data-f="tcp"]', card).textContent = `TCP 보정: ${tcp.measured ? tcp.path : "identity (측정 전)"}`;
  tail($('[data-f="tail"]', card), u);
}

function renderHead(s) {
  if (!s.station.has_head) return;
  const card = $("#card-head");
  const u = s.units.head;
  const h = s.head;
  let override = null;
  if (h && running("head") && !stopping("head")) {
    override = h.returning ? ["live", "home 으로 복귀 중"] : h.locked ? ["ok", "잠김 (home): Space 로 따라가기"]
      : h.state === "hold" ? ["warn", "HMD 놓침: 멈춤"] : ["live", "HMD 따라가는 중"];
  }
  setBadge($('[data-f="phase"]', card), unitBadge(u, override));
  card.classList.toggle("running", running("head"));
  card.classList.toggle("stopping", stopping("head"));
  const rows = [];
  if (h) {
    const dp = h.meas_pan_deg - h.home_pan_deg, dt = h.meas_tilt_deg - h.home_tilt_deg;
    rows.push(["pan (home 기준)", `${fmt(dp)}°  [오른 ${fmt(h.window_deg?.[0], 0)} / 왼 ${fmt(h.window_deg?.[1], 0)}]`]);
    rows.push(["tilt (home 기준)", `${fmt(dt)}°  [아래 ${fmt(h.window_deg?.[2], 0)} / 위 ${fmt(h.window_deg?.[3], 0)}]`]);
    rows.push(["HMD", h.hmd_tracked ? "추적 중" : "놓침", h.hmd_tracked ? "ok" : "warn"]);
  } else rows.push(["상태", running("head") ? "기록 기다림" : "꺼짐"]);
  facts($('[data-f="facts"]', card), rows);
  tail($('[data-f="tail"]', card), u, 4);
}

function renderRecord(s) {
  const card = $("#card-record");
  const u = s.units.record;
  setBadge($('[data-f="phase"]', card), unitBadge(u));
  card.classList.toggle("running", running("record"));
  card.classList.toggle("stopping", stopping("record"));
  syncInput("#task", s.settings.task);
  syncInput("#episodes", s.settings.episodes);
  const sc = s.station.sidecars || {};
  const streams = Object.keys(sc).filter((n) => running(n === "head" ? "head" : n));
  $('[data-f="streams"]', card).textContent = running("record") ? "" :
    `시작하면 필수 스트림: ${streams.length ? streams.join(", ") : "없음 (팔만, --no-sidecars)"} · 팔 ${({ right: "오른팔", left: "왼팔", both: "양팔" })[s.settings.arm_side]}`;
  tail($('[data-f="tail"]', card), u);
}

function renderGlove(s) {
  if (!s.station.hands.length) return;
  const card = $("#card-glove");
  const g = (s.probes || {}).gloves || {};
  const ros = (s.probes || {}).ros || {};
  setBadge($('[data-f="sensecom"]', card), g.sensecom_started_at ? ["ok", "SenseCom 실행 중"] : ["mute", "SenseCom 꺼짐"]);
  const rows = [];
  for (const [side, glove] of Object.entries(g.gloves || {})) {
    const topic = (ros.glove_topics || {})[side];
    rows.push([`${SIDE_KO[side]}손 ${glove.serial}`, `${glove.connected ? "BLE 연결" : "BLE 미연결"} · 토픽 ${topic ? "있음" : "없음"}`, glove.connected && topic ? "ok" : "warn"]);
    const cal = (s.calibration || {})[side];
    if (cal) rows.push([`${SIDE_KO[side]}손 보정`, cal.ok ? "유효" : cal.detail, cal.ok ? "ok" : "warn"]);
  }
  rows.push(["드라이버", (g.driver_pids || []).length ? `PID ${g.driver_pids.join(", ")}` : "꺼짐", (g.driver_pids || []).length ? "ok" : ""]);
  facts($('[data-f="facts"]', card), rows);
  syncInput("#user", s.settings.user);
  const calib = ["calib_right", "calib_left"].find(running);
  const enter = $("[data-key-active]", card);
  enter.disabled = !calib;
  enter.dataset.key = calib || "";
  const fakeOnly = s.mode !== "real";
  $$('[data-action^="glove:"]', card).forEach((b) => { b.disabled = fakeOnly; b.title = fakeOnly ? "실기 모드에서만" : ""; });
  $('[data-action="glove_driver_stop"]', card).disabled = !(g.driver_pids || []).length;
  tail($('[data-f="tail"]', card), s.units[calib || "calib_right"], 5);
}

function renderHands(s) {
  if (!s.station.hands.length) return;
  const card = $("#card-hands");
  const ros = (s.probes || {}).ros || {};
  $('[data-f="domain"]', card).textContent = ros.error ? `ROS: ${ros.error}` : `ROS 도메인 ${ros.domain ?? "-"} · ${ros.took_s ?? "-"} s`;
  const box = $('[data-f="rows"]', card);
  if (!box.dataset.built) {
    box.innerHTML = s.station.hands.map((side) => `<section class="hand-row" data-hand="${side}">
      <header><b>${SIDE_KO[side]}손</b><span class="badge" data-f="state">-</span></header>
      <dl class="facts" data-f="facts"></dl>
      <div class="actions">
        <button type="button" class="btn" data-start="hand_${side}">노드 시작</button>
        <button type="button" class="btn primary" data-action="hand_on:${side}">켜기 (장갑 따라가기)</button>
        <button type="button" class="btn" data-action="hand_off:${side}">끄기 (펼침)</button>
        <button type="button" class="btn danger" data-stop="hand_${side}">정지</button>
        <button type="button" class="btn ghost" data-log="hand_${side}">로그</button>
      </div></section>`).join("");
    box.dataset.built = "1";
  }
  for (const side of s.station.hands) {
    const row = $(`[data-hand="${side}"]`, box);
    const key = `hand_${side}`;
    const h = (ros.hands || {})[side] || {};
    const rec = (s.hands || {})[side];
    let override = null;
    if (rec && running(key) && !stopping(key))
      override = rec.fault ? ["bad", `FAULT ${rec.fault}`] : [rec.mode === "disabled" ? "ok" : "live", `${rec.mode} · ${rec.state}`];
    setBadge($('[data-f="state"]', row), unitBadge(s.units[key], override));
    const rows = [["EtherCAT 드라이버", h.driver ? "있음" : "없음 (s2r 콘솔에서 켤 것)", h.driver ? "ok" : "warn"],
      ["angle_set 발행자", String(h.angle_set_publishers ?? "-"), (h.angle_set_publishers || 0) > (running(key) ? 1 : 0) ? "bad" : ""]];
    if (rec) {
      rows.push(["장갑 나이", `${fmt(rec.glove_age_s, 2)} s${rec.glove_frozen ? " (멈춤)" : ""}`, rec.glove_frozen ? "warn" : ""]);
      if (rec.refusal) rows.push(["켜기 거부", rec.refusal, "bad"]);
    }
    facts($('[data-f="facts"]', row), rows);
    $(`[data-action="hand_on:${side}"]`, row).disabled = !running(key) || stopping(key);
    $(`[data-action="hand_off:${side}"]`, row).disabled = !running(key) || stopping(key);
  }
}

function renderBar(s) {
  const chips = s.running.map((k) => `<button type="button" class="chip ${stopping(k) ? "stopping" : ""}" data-log="${k}">${esc(unitName(k))}</button>`).join("");
  if ($("#running").dataset.html !== chips) { $("#running").innerHTML = chips; $("#running").dataset.html = chips; }
  const sel = $("#space-target");
  if (document.activeElement !== sel) sel.value = s.space_target || "";
  $$("[data-key]").forEach((b) => b.classList.toggle("target", b.dataset.key === s.space_target && b.dataset.text === " "));
  $("#stop-all").disabled = s.running.length === 0;
}

const EXCLUDES = { arm: "record", record: "arm", quest_view: "mock_quest", mock_quest: "quest_view" };

function renderButtons() {
  $$("[data-start]").forEach((b) => {
    const other = EXCLUDES[b.dataset.start];
    b.disabled = running(b.dataset.start) || (other && running(other));
    b.title = other && running(other) ? `${unitName(other)} 실행 중` : "";
  });
  $$("[data-stop]").forEach((b) => { b.disabled = !running(b.dataset.stop) || stopping(b.dataset.stop); });
  $$("[data-key]:not([data-key-active])").forEach((b) => { b.disabled = !running(b.dataset.key) || stopping(b.dataset.key); });
}

function syncInput(sel, value) {
  const el = $(sel);
  if (document.activeElement !== el && el.value !== String(value ?? "")) el.value = value ?? "";
}

function renderEvents(s) {
  for (const e of s.events || []) {
    if (e.t <= lastEventT) continue;
    lastEventT = e.t;
    if (e.level !== "info") toast(e.text, e.level === "bad" ? "bad" : "");
  }
}

function render(s) {
  S = s;
  renderTop(s);
  renderHealth(s);
  renderAlerts(s);
  renderQuest(s);
  renderArm(s);
  renderHead(s);
  renderRecord(s);
  renderGlove(s);
  renderHands(s);
  renderBar(s);
  renderButtons();
  renderEvents(s);
}

// ---------------------------------------------------------------- drawer
async function refreshDrawer() {
  if (!drawerKey) return;
  try {
    const r = await (await fetch(`/api/log?key=${encodeURIComponent(drawerKey)}&lines=400`)).json();
    const pre = $("#drawer-log");
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 20;
    const text = (r.lines || []).join("\n");
    if (pre.textContent !== text) { pre.textContent = text; if (atBottom) pre.scrollTop = pre.scrollHeight; }
    $("#drawer-path").textContent = r.path || "";
  } catch { /* the next tick retries */ }
  $("#drawer-force").disabled = !running(drawerKey);
}

function openDrawer(key) {
  drawerKey = key;
  $("#drawer-title").textContent = `로그: ${unitName(key)}`;
  $("#drawer-log").textContent = "";
  $("#drawer").hidden = false;
  refreshDrawer();
}

// ---------------------------------------------------------------- events
document.addEventListener("click", async (ev) => {
  const b = ev.target.closest("button");
  if (!b) return;
  const d = b.dataset;
  if (d.mode && !b.disabled && S && d.mode !== S.mode) {
    if (d.mode === "real" && !(await confirmDialog("실기 모드로", "이제부터 시작하는 것은 실제 로봇·장치를 쓴다. 움직이는 단위는 시작 전에 명령을 다시 보여 준다.", "", true))) return;
    act("mode", { mode: d.mode }, null, d.mode === "real" ? "실기 모드" : "FAKE 모드");
  } else if (d.start) act("start", { key: d.start }, b, `${unitName(d.start)} 시작`);
  else if (d.stop) act("stop", { key: d.stop }, b, `${unitName(d.stop)} 정지 요청 (안전 자세로)`);
  else if (d.key !== undefined && d.text !== undefined && d.key) act("key", { key: d.key, text: d.text }, null, `${unitName(d.key)} ← ${d.text === " " ? "Space" : d.text.trim() || "Enter"}`);
  else if (d.send) act("key", { key: d.send, text: d.text }, b, "입력 보냄");
  else if (d.action) act("action", { name: d.action }, b, "요청함");
  else if (d.toggle) act(running(d.toggle) ? "stop" : "start", { key: d.toggle }, b);
  else if (d.side) act("settings", { arm_side: d.side }, null, `팔: ${b.textContent}`);
  else if (d.log) openDrawer(d.log);
  else if (d.left) act("action", { name: `left_behind_stop:${d.left}` }, b, "SIGINT 보냄");
});

for (const [sel, field, cast] of [["#scale", "scale", Number], ["#task", "task", String], ["#episodes", "episodes", Number], ["#user", "user", String]]) {
  $(sel).addEventListener("change", (ev) => act("settings", { [field]: cast(ev.target.value) }, null, "설정 저장"));
}
$("#space-target").addEventListener("change", (ev) => act("space_target", { key: ev.target.value || null }, null, "Space 대상 바꿈"));
$("#stop-all").addEventListener("click", async (ev) => {
  if (await confirmDialog("모두 정지", "실행 중인 모든 단위에 SIGINT 를 한 번 보낸다. 팔은 home → 차렷, 머리는 home, 손은 펼침으로 간 뒤 끝난다.", "", false))
    act("stop_all", {}, ev.currentTarget, "모두 정지 요청");
});
$("#quit").addEventListener("click", async () => {
  if (!(await confirmDialog("콘솔 종료", "실행 중인 것을 모두 안전 자세로 정지한 뒤 콘솔 서버를 내린다. 창만 닫으려면 창을 닫으면 된다(서버와 로봇은 그대로).", "", false))) return;
  const r = await post("quit", { confirm: true });
  toast(r.ok ? "콘솔 종료 중: 안전 자세 복귀 뒤 서버가 내려간다" : r.error, r.ok ? "ok" : "bad");
});
$("#theme").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("macq-theme", next);
});
$("#drawer-close").addEventListener("click", () => { $("#drawer").hidden = true; drawerKey = null; });
$("#drawer-force").addEventListener("click", async () => {
  if (!drawerKey) return;
  if (await confirmDialog(`${unitName(drawerKey)} 강제 종료`, "SIGTERM 뒤 5 s 안에 안 끝나면 SIGKILL. 안전 자세 복귀가 없다: 팔이면 받친 뒤에만.", "", true))
    act("force", { key: drawerKey, confirm: true }, null, "강제 종료 보냄");
});
$("#drawer-input").addEventListener("submit", (ev) => {
  ev.preventDefault();
  if (!drawerKey) return;
  act("key", { key: drawerKey, text: $("#drawer-text").value + "\n" }, null, "입력 보냄");
  $("#drawer-text").value = "";
});
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && !$("#drawer").hidden && !$("#confirm").open) { $("#drawer").hidden = true; drawerKey = null; return; }
  if (ev.code !== "Space" || ev.repeat || $("#confirm").open) return;
  if (ev.target.closest("input, textarea, select")) return;
  ev.preventDefault();  // Space goes to the robot program, not to the focused button
  if (!S?.space_target) { toast("Space 대상 없음: 아래 막대에서 고른다", "bad"); return; }
  act("key", { key: S.space_target, text: " " }, null, `${unitName(S.space_target)} ← Space`);
});

// ---------------------------------------------------------------- start
(function init() {
  const saved = new URLSearchParams(location.search).get("theme") || localStorage.getItem("macq-theme");
  document.documentElement.dataset.theme = saved || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  if (new URLSearchParams(location.search).has("once")) {  // one render, no stream (headless screenshots)
    fetch("/api/state").then((r) => r.json()).then((s) => { setBadge($("#conn"), ["ok", "한 번 읽음"]); render(s); });
    return;
  }
  const es = new EventSource("/api/stream");
  es.addEventListener("state", (ev) => {
    setBadge($("#conn"), ["ok", "연결됨"]);
    render(JSON.parse(ev.data));
  });
  es.onerror = () => setBadge($("#conn"), ["bad", "콘솔 서버 끊김"]);
  setInterval(refreshDrawer, 1000);
})();
