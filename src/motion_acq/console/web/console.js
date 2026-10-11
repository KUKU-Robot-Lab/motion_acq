"use strict";
// macq console window: a body-shaped map of the devices; each node has one main action,
// everything else is in the detail panel on the right. State comes from /api/stream.

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
let S = null;
let selected = "quest";
let drawerKey = null;
let lastEventT = Date.now() / 1000;
const SIDE_KO = { right: "오른", left: "왼" };
const UNIT_KO = { quest_view: "영상 서버", mock_quest: "가짜 Quest", head: "목", arm: "팔 전체", record: "녹화",
  arm_right: "로봇 오른팔", arm_left: "로봇 왼팔",
  hand_right: "로봇 오른손", hand_left: "로봇 왼손", calib_right: "오른손 보정", calib_left: "왼손 보정",
  rezero_right: "오른손 편 손 맞춤", rezero_left: "왼손 편 손 맞춤",
  fingers_right: "오른손 손가락별 맞춤", fingers_left: "왼손 손가락별 맞춤",
  tune_right: "오른손 손가락 튜닝", tune_left: "왼손 손가락 튜닝",
  ecat_right: "로봇 오른손 드라이버", ecat_left: "로봇 왼손 드라이버" };
const CALIB_KEYS = ["calib_right", "calib_left", "rezero_right", "rezero_left", "fingers_right", "fingers_left"];
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
  try { return await r.json(); } catch { return { ok: false, error: `콘솔 서버 응답을 읽지 못했다 (HTTP ${r.status})` }; }
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

function setBadge([tone, text]) {
  const el = $("#conn");
  el.className = `badge ${tone}`;
  el.textContent = text;
}

// ---------------------------------------------------------------- helpers
const unit = (k) => S?.units?.[k] || {};
const running = (k) => !!unit(k).running;
const stopping = (k) => running(k) && unit(k).stopping_s !== null && unit(k).stopping_s !== undefined;
const job = (name) => (S?.jobs || {})[name] || {};
const jobRunning = (name) => job(name).state === "running";
const real = () => S?.mode === "real";
const armSides = () => ({ right: ["right"], left: ["left"], both: ["right", "left"] })[S.settings.arm_side] || [];
// the all-at-once units (arm / record) move the arm_side arms; arm_right / arm_left one arm on its own (10.11 user)
const allArmOn = (side) => ["arm", "record"].find((k) => running(k) && armSides().includes(side));
const oneArmOn = () => ["arm_right", "arm_left"].some((k) => running(k) || jobRunning(k));

function btn(label, attrs, cls = "") {
  const a = Object.entries(attrs).map(([k, v]) => (v === true ? k : v === false || v == null ? "" : `${k}="${esc(v)}"`)).join(" ");
  return `<button type="button" class="btn ${cls}" ${a}>${label}</button>`;
}

function unitState(k, fallback) {
  const u = unit(k);
  if (!u.running && (u.rc === null || u.rc === undefined)) return null;
  if (stopping(k)) return ["warn", u.forced ? `강제 종료 중 ${fmt(u.stopping_s, 0)} s` : `정지 중 ${fmt(u.stopping_s, 0)} s (안전 자세로)`];
  if (u.running) return fallback || u.phase || ["live", "실행 중"];
  if (u.rc === 0) return null;
  return ["bad", `끝남 (rc ${u.rc})${u.phase && u.phase[0] === "bad" ? ": " + u.phase[1] : ""}`];
}

function jobLine(name) {
  const j = job(name);
  if (!j.state) return "";
  return j.state === "running" ? "진행 중..." : j.text;
}

// ---------------------------------------------------------------- node models
// Each returns {title, dev, tone, state, sub, actions}; actions is HTML (one main button, sometimes two).
function nodeQuest(s) {
  const q = (s.probes || {}).quest || {};
  const usb = (q.devices || []).length === 1 && q.devices[0] === "device";
  const qv = s.quest_view;
  const n = { title: "머리", dev: "Quest 3", actions: "" };
  const connect = btn("연결", { "data-action": "quest:connect" }, "primary");
  // 10.11 user: set up without wearing the headset (proximity sensor off + wake, then connect + VR)
  const offhead = btn("안 쓰고 시작", { "data-action": "quest:offhead", title: "근접 센서를 끄고 헤드셋을 깨운 뒤 연결 + VR 시작 (되돌리기: 자세히 > 근접 센서 원래대로)" }, "ghost");
  if (jobRunning("quest")) return { ...n, tone: "live", state: "연결 중...", sub: "" };
  if (!real()) {
    if (running("mock_quest")) return { ...n, tone: "ok", state: "가짜 Quest", sub: running("quest_view") ? "테스트 영상 송출" : "" };
    return { ...n, tone: "mute", state: "꺼짐", sub: "가짜 Quest + 테스트 영상", actions: connect };
  }
  if (s.quest_source === "app") return { ...n, tone: "ok", state: "HandUMI 앱 (영상 없음)", sub: "영상으로 바꾸려면 [연결]", actions: connect };
  if (s.quest_source === "external") return { ...n, tone: "bad", state: "콘솔 밖 프로그램이 65432 사용", sub: "자세히 보기" };
  if (!running("quest_view")) {
    return { ...n, tone: usb ? "mute" : "bad", state: usb ? "꺼짐" : (q.text || "확인 중"),
             sub: q.awake === false ? "헤드셋 잠듦: 쓰면 깬다" : jobLine("quest"), actions: connect + (usb ? offhead : "") };
  }
  const last = qv ? parseFloat(qv.last) : NaN;
  if (qv && isFinite(last) && last < 3) {
    return { ...n, tone: "ok", state: "영상 송출 중", sub: `자세 받는 곳 ${qv.clients} · 프레임 ${qv.frames}` };
  }
  return { ...n, tone: "warn", state: "페이지 열림: 헤드셋을 쓰고 [시작]", sub: jobLine("quest"),
           actions: btn("시작", { "data-action": "quest:start" }, "go") + offhead };
}

function nodeNeck(s) {
  const n = { title: "목", dev: "다이나믹셀 XC330", sub: "" };
  if (!running("head")) {
    const st = unitState("head");
    return { ...n, tone: st ? st[0] : "mute", state: st ? st[1] : "꺼짐", actions: btn("시작", { "data-start": "head" }, "primary") };
  }
  const h = s.head;
  let st = ["live", "기록 기다림"];
  if (h) st = h.returning ? ["live", "home 으로 복귀 중"] : h.locked ? ["ok", "잠김 (home): Space 로 따라가기"]
    : h.state === "hold" ? ["warn", "HMD 놓침: 멈춤"] : ["live", "머리 따라가는 중"];
  st = unitState("head", st);
  const sub = h ? `pan ${fmt(h.meas_pan_deg - h.home_pan_deg)}° · tilt ${fmt(h.meas_tilt_deg - h.home_tilt_deg)}°` : "";
  return { ...n, tone: st[0], state: st[1], sub,
           actions: btn("Space", { "data-key": "head", "data-text": " " }, "key small") +
                    btn("정지", { "data-stop": "head" }, "danger small") };
}

function nodeGlove(s, side) {
  const g = ((s.probes || {}).gloves || {}).gloves?.[side] || {};
  const topic = (((s.probes || {}).ros || {}).glove_topics || {})[side];
  const cal = (s.calibration || {})[side];
  const n = { title: `${SIDE_KO[side]}손`, dev: `Nova 2 ${g.serial || ""}` };
  if (!real()) return { ...n, tone: "mute", state: "fake: 로봇 손을 켜면 가짜 장갑", sub: cal ? (cal.ok ? "보정 있음" : "보정 없음") : "" };
  if (!s.sensecom_installed) return { ...n, tone: "bad", state: "SenseCom 미설치", sub: "이 PC: scripts/ros_ws_setup.sh --full" };
  if (jobRunning("glove")) return { ...n, tone: "live", state: "연결 중...", sub: "" };
  const connect = btn("연결", { "data-action": "glove:connect" }, "primary");
  if (g.connected && topic && cal && cal.sdk) {  // direct: the SDK's own calibration is the only one (10.11)
    const recal = btn(cal.ok ? "장갑 보정 다시" : "장갑 보정", { "data-action": `glove_recal:${side}`,
      title: "SDK 보정을 처음부터: 음성 안내에 따라 다섯 손가락을 끝까지 쥐었다 펴고 엄지를 크게 돌린다" }, cal.ok ? "ghost small" : "primary");
    return { ...n, tone: cal.ok ? "ok" : "warn", state: cal.ok ? "연결됨 · 장갑 보정 완료" : "연결됨 · 장갑 보정 필요",
             sub: cal.ok ? "" : cal.detail.replace("장갑 SDK 보정: ", ""), actions: recal };
  }
  if (g.connected && topic) {
    const calOk = cal && cal.ok;
    return { ...n, tone: calOk ? "ok" : "warn", state: calOk ? "연결됨 · 보정 유효" : "연결됨 · 보정 필요",
             sub: calOk ? "" : "보정 사용자 이름을 넣고 손마다 보정한다",
             actions: calOk ? "" : btn("보정", { "data-select": `glove_${side}` }, "primary") };
  }
  if (g.connected) return { ...n, tone: "warn", state: "BLE 연결, 드라이버 없음", sub: jobLine("glove"), actions: connect };
  return { ...n, tone: "mute", state: "미연결", sub: jobLine("glove") || "장갑 전원을 켜고 [연결]", actions: connect };
}

// 10.11 user: a failed return must not end in a drop. The arm holds and asks: h / r / Enter alone.
const recovering = (key) => /\[복구\]/.test(unit(key).prompt || "");
function recoveryButtons(key) {
  return btn("다시 홈", { "data-send": key, "data-text": "h\n", title: "멈춘 자리에서 천천히 home 으로 가서 버틴다 (분석용)" }, "primary") +
         btn("차렷", { "data-send": key, "data-text": "r\n", title: "home 에서 저장 경로로 차렷 (손은 주먹) → 모터 끔" }, "") +
         btn("모터 끄기", { "data-send": key, "data-text": "\n", "data-confirm": "팔을 손으로 받쳤나? 모터가 꺼지면 팔이 떨어진다.",
                            title: "팔을 받친 뒤: 그 자리에서 모터를 끈다" }, "danger small");
}

function nodeArm(s, side) {
  const key = `arm_${side}`;
  const can = ((s.probes || {}).can || {})[side] || {};
  const n = { title: `로봇 ${SIDE_KO[side]}팔`, dev: `OpenArm · ${can.port || "-"}` };
  const canText = can.up && can.fd ? "CAN UP FD" : can.exists === false ? "CAN 없음" : can.up ? "CAN FD 아님" : "CAN DOWN";
  const all = allArmOn(side);
  if (all) {
    const st = unitState(all) || ["live", "실행 중"];
    return { ...n, tone: st[0], state: st[1], sub: `${unitName(all)}로 움직이는 중 (가운데 버튼)` };
  }
  if (jobRunning(key)) return { ...n, tone: "live", state: "손 준비 중...", sub: canText };
  if (!running(key)) {
    const st = unitState(key);
    const failed = job(key).state === "failed" ? job(key).text : unit(key).error ? `지난 시작 실패: ${unit(key).error}` : "";
    const busy = running("arm") || running("record");
    return { ...n, tone: st ? st[0] : failed ? "bad" : (can.up && can.fd || !real() ? "ok" : "warn"),
             state: st ? st[1] : "꺼짐 (차렷)", sub: failed || canText,
             actions: btn("홈", { "data-start": key, disabled: busy, title: busy ? "전체(팔·녹화)가 실행 중" :
               "차렷 → 저장 경로 → home. 손은 주먹 그대로(장갑·Quest 필요 없음). 이 팔만 움직인다" }, "primary") };
  }
  const st = unitState(key) || ["live", "실행 중"];
  if (recovering(key))
    return { ...n, tone: "bad", state: (unit(key).phase || [, "멈춤"])[1], sub: "팔은 그 자리에서 버티는 중 (정지 중이어도 고를 수 있다)",
             actions: recoveryButtons(key) };
  return { ...n, tone: st[0], state: st[1], sub: canText,
           actions: btn("따라가기", { "data-key": key, "data-text": " ", title: "Space: 이 팔이 컨트롤러를 따라간다 (Quest 연결 필요)" }, "key") +
                    btn("홈", { "data-key": key, "data-text": "h", title: "H: 따라가던 팔이 home 으로 돌아가 Space 를 기다린다" }, "key") +
                    btn("차렷", { "data-stop": key, title: "home → 차렷(저장 경로, 손은 주먹) → 모터 끔" }, "danger small") };
}

function driverText(s, side) {
  if (!real()) return "fake: 켜면 가짜 드라이버·장갑과 함께 뜬다";
  const ext = ((s.probes || {}).hand_drivers || {})[side] || [];
  const h = (((s.probes || {}).ros || {}).hands || {})[side] || {};
  if (running(`ecat_${side}`)) return h.driver ? "드라이버 켜짐 (이 콘솔)" : "드라이버 올라오는 중";
  if (ext.length) return `드라이버: s2r 쪽 (PID ${ext.join(",")})`;
  return h.driver ? "드라이버 있음" : "드라이버는 [켜기] 때 같이 켠다";
}

const HAND_PHASE = { to_home: ["live", "home(펼침)으로 이동"], reference: ["warn", "기준 자세: 로봇 손 home 과 같은 모양(엄지·검지 C자), 1초 멈춤"],
  follow: ["live", "장갑 따라가는 중"], rest: ["ok", "주먹 (팔이 경로를 도는 중)"], return_home: ["live", "주먹으로 닫는 중"] };

function nodeHand(s, side) {
  const key = `hand_${side}`;
  const rec = (s.hands || {})[side];
  const n = { title: `로봇 ${SIDE_KO[side]}손`, dev: "RH56F1 EtherCAT" };
  const manual = running(key) ? rec?.manual : null;
  const home = btn("홈", { "data-action": `hand:${side}:home`, title: "home(펼침)으로 가서 멈춘다(장갑 무시)" }, manual === "home" ? "go" : "primary");
  const follow = btn("따라가기", { "data-action": `hand:${side}:follow`, title: "home → 1초 기준 자세 → 장갑을 따라간다(팔과 무관, 장갑 필요)" },
                     manual === "follow" ? "go" : "primary");
  const fist = btn("주먹", { "data-action": `hand:${side}:fist`, title: "주먹으로 쥐고 멈춘다(노드는 그대로, 팔이 home 에 와도 안 폄)" },
                   manual === "fist" ? "go" : "");
  if (jobRunning(key)) return { ...n, tone: "live", state: "준비 중 (드라이버·노드)...", sub: "" };
  if (!running(key)) {
    const st = unitState(key);
    const failed = job(key).state === "failed" ? job(key).text : "";
    return { ...n, tone: st ? st[0] : failed ? "bad" : "mute", state: st ? st[1] : "꺼짐",
             sub: failed || driverText(s, side), actions: home + follow + fist };
  }
  let st = ["ok", "노드 실행 (손 꺼짐)"];
  if (rec?.fault) st = ["bad", `FAULT ${rec.fault}`];
  else if (rec?.mode === "enabled" && rec.phase === "wait_arm")
    st = manual === "home" ? ["ok", "home 에서 멈춤"] : ["ok", "home 대기: 팔이 따라가면 같이 따라감"];
  else if (rec?.mode === "enabled" && rec.phase === "rest" && manual === "fist") st = ["ok", "주먹으로 멈춤"];
  else if (rec?.mode === "enabled" && rec.phase === "follow" && (rec.glove_age_s == null || rec.glove_frozen))
    // 10.11: the left glove driver had died; the hand sat in follow and held still
    st = ["warn", `장갑 데이터 없음: 손이 멈춰 있다 (장갑 [연결] 다시)`];
  else if (rec?.mode === "enabled") st = HAND_PHASE[rec.phase] || ["live", rec.phase || "켜짐"];
  st = unitState(key, st);
  const how = manual === "home" ? "수동: 홈" : manual === "follow" ? "수동: 따라가기" : manual === "fist" ? "수동: 주먹" : "자동: 팔과 같이";
  const sub = job(key).state === "failed" ? job(key).text : rec?.refusal || rec?.reference || how;
  return { ...n, tone: st[0], state: st[1], sub,
           actions: home + follow + fist +
                    btn("끄기", { "data-action": `hand_off:${side}`, title: "주먹으로 쥔 뒤 노드를 내린다 (팔과 무관; 팔 [차렷] 전에 다시 [주먹])" }, "danger small") };
}

function nodeRecord(s) {
  const items = s.record_checklist || [];
  const chips = `<div class="checklist">${items.map((i) => `<span class="check ${i.ok ? "ok" : "miss"}" title="${esc(i.detail)}">${i.ok ? "✓" : "!"} ${esc(i.name)}</span>`).join("")}</div>`;
  const n = { title: "녹화", dev: "LeRobot 데이터셋 · 모든 스트림 + 영상" };
  if (!running("record")) {
    const st = unitState("record");
    const armOn = running("arm");
    return { ...n, tone: st ? st[0] : "mute", state: st ? st[1] : "대기", sub: "", body: chips,
             actions: btn(`<svg><use href="#i-rec"/></svg>녹화 시작`, { "data-start": "record", disabled: armOn,
                          title: armOn ? "팔 원격조작을 먼저 정지" : "" }, "danger big") };
  }
  const st = unitState("record") || ["live", "실행 중"];
  return { ...n, tone: st[0], state: st[1], sub: "", body: chips,
           actions: btn("Space 시작/저장", { "data-key": "record", "data-text": " " }, "key") +
                    btn("R 다시", { "data-key": "record", "data-text": "r" }, "key") +
                    btn("Q 마치기", { "data-key": "record", "data-text": "q" }, "key") +
                    btn("정지", { "data-stop": "record" }, "danger") };
}

function nodeHtml(m) {
  return `<div class="n-head"><span class="dot"></span><span class="n-title">${esc(m.title)}</span><span class="n-dev">${esc(m.dev)}</span></div>
    <div class="n-state">${esc(m.state)}</div><div class="n-sub">${esc(m.sub || "")}</div>${m.body || ""}
    <div class="n-actions">${m.actions || ""}</div>`;
}

const NODES = {
  quest: nodeQuest, neck: nodeNeck,
  glove_left: (s) => nodeGlove(s, "left"), glove_right: (s) => nodeGlove(s, "right"),
  arm_left: (s) => nodeArm(s, "left"), arm_right: (s) => nodeArm(s, "right"),
  hand_left: (s) => nodeHand(s, "left"), hand_right: (s) => nodeHand(s, "right"),
  record: nodeRecord,
};

function nodeHidden(s, id) {
  if (id === "neck") return !s.station.has_head;
  if (/^(glove|hand)_/.test(id)) return !s.station.hands.includes(id.split("_")[1]);
  return false;
}

function renderNodes(s) {
  for (const el of $$(".node")) {
    const id = el.dataset.node;
    el.hidden = nodeHidden(s, id);
    if (el.hidden) continue;
    const m = NODES[id](s);
    el.className = `node${id === "record" ? " rec" : ""} tone-${m.tone}${id === selected ? " selected" : ""}`;
    const html = nodeHtml(m);
    if (el.dataset.html !== html && !el.contains(document.activeElement)) { el.innerHTML = html; el.dataset.html = html; }
  }
  renderArmCtrl(s);
}

function renderArmCtrl(s) {
  // all at once (10.11 user: after each arm and hand was checked on its own): both arms + both hands
  const box = $("#arm-ctrl");
  let html;
  if (running("arm")) {
    html = `<span class="meta">전체</span>` +
           btn("전체 따라가기", { "data-key": "arm", "data-text": " ", title: "Space: 팔과 손이 같이 따라간다" }, "key") +
           btn("전체 홈", { "data-key": "arm", "data-text": "h", title: "H: 따라가던 팔이 home 으로, 손은 펼침" }, "key") +
           btn("정지 (home → 차렷)", { "data-stop": "arm" }, "danger") +
           `<span class="meta">${esc((unitState("arm") || ["", ""])[1])}</span>`;
  } else {
    const err = unit("arm").error;
    const block = running("record") ? "녹화가 팔을 쓰는 중" : oneArmOn() ? "한 팔씩 실행 중: 그 팔을 먼저 정지" : "";
    html = btn(`<svg><use href="#i-play"/></svg>전체 시작`, { "data-start": "arm", "data-opts": JSON.stringify({ arm_side: "both" }),
                disabled: !!block || jobRunning("arm"), title: block || "양팔 + 양손: 차렷 → home, 손은 자동(팔이 따라가면 같이)" }, "primary big") +
           (err ? `<span class="meta" style="color: var(--bad)">지난 시작 실패: ${esc(err)}</span>`
                : `<span class="meta">${esc(block || `양팔+양손 · Space 로 같이 따라가기 · 배율 ${fmt(s.settings.scale, 1)}`)}</span>`);
  }
  if (box.dataset.html !== html) { box.innerHTML = html; box.dataset.html = html; }
}

// ---------------------------------------------------------------- detail panel
function facts(rows) {
  return `<dl class="facts">${rows.map(([k, v, tone]) => `<dt>${esc(k)}</dt><dd class="${tone || ""}">${esc(v)}</dd>`).join("")}</dl>`;
}

const logButton = (key) => btn("전체 로그", { "data-log": key }, "ghost small");

function directionHtml(s) {
  const d = s.direction || {};
  const ok = s.settings.webxr_arm_ok;
  const status = ok
    ? `<p class="hint">확인됨 ${esc(ok)}: 헤드셋 영상 모드로 팔을 움직일 수 있다. ${btn("다시 확인 필요로", { "data-action": "direction:reset" }, "ghost small")}</p>`
    : `<p class="hint">아직 확인 전: 그동안 헤드셋 영상 모드에서는 팔 시작이 막힌다.</p>`;
  if (!d.active) return status + `<div class="actions">${btn("방향 확인 켜기", { "data-action": "direction:on" }, "small")}</div>`;
  const side = (k) => d.sides?.[k]
    ? `<b>${esc(d.sides[k].main)}</b><span class="meta">앞 ${d.sides[k].x_cm} · 왼 ${d.sides[k].y_cm} · 위 ${d.sides[k].z_cm} cm</span>`
    : `<b>-</b><span class="meta">${d.tracked?.[k] ? "" : "안 보임"}</span>`;
  return status + `<div class="readout"><span>왼손 컨트롤러</span><div>${side("left")}</div><span>오른손 컨트롤러</span><div>${side("right")}</div></div>
    <p class="hint">${esc(d.waiting || d.error || "컨트롤러를 앞으로 → '앞', 왼쪽 → '왼', 위로 → '위' 가 나오면 맞다. 로봇은 움직이지 않는다.")}</p>
    <div class="actions">${btn("기준 다시", { "data-action": "direction:rebase" }, "small")}${btn("방향 맞음: 팔 허용", { "data-action": "direction:ok" }, "go small")}${btn("끄기", { "data-action": "direction:off" }, "ghost small")}</div>`;
}

function detailQuest(s) {
  const q = (s.probes || {}).quest || {};
  const qv = s.quest_view;
  const owner = (s.probes || {}).quest_port;
  const rows = [["USB", q.text || "확인 중", (q.devices || []).length === 1 && q.devices[0] === "device" ? "ok" : "bad"],
    ["헤드셋", q.awake === undefined || q.awake === null ? "-" : q.awake ? "깨어 있음" : "잠듦 (쓰면 깬다)", q.awake ? "ok" : "warn"],
    ["자세 입력", ({ app: "HandUMI 앱", view: "헤드셋 영상(WebXR)", mock: "가짜 Quest", external: "콘솔 밖 프로그램" })[s.quest_source] || "없음"],
    ["영상 서버", running("quest_view") ? "실행 중" : "꺼짐", running("quest_view") ? "live" : ""]];
  if (qv && real()) rows.push(["헤드셋 자세", `${qv.poses} 개 (마지막 ${qv.last} 전) · 받는 곳 ${qv.clients}`]);
  if (s.quest_source === "external" && owner) rows.push(["TCP 65432", `콘솔 밖 ${owner.name} (PID ${owner.pid ?? "?"})`, "bad"]);
  if (jobLine("quest")) rows.push(["최근", jobLine("quest"), job("quest").state === "failed" ? "bad" : ""]);
  const extra = real()
    ? btn("VR 다시 시작", { "data-action": "quest:start" }) + btn("페이지 다시 열기", { "data-action": "quest:connect" }) +
      btn("영상 끊기", { "data-stop": "quest_view" }, "ghost") + btn("HandUMI 앱 모드", { "data-action": "quest:app" }, "ghost") +
      btn("헤드셋 벗고 쓰기", { "data-action": "quest:unworn" }, "ghost") + btn("근접 센서 원래대로", { "data-action": "quest:worn" }, "ghost")
    : btn("가짜 Quest 정지", { "data-stop": "mock_quest" }, "ghost") + btn("테스트 영상 정지", { "data-stop": "quest_view" }, "ghost");
  return `<h3>머리 · Quest 3</h3>${facts(rows)}<div class="actions">${extra}</div>
    <div class="sect">방향 확인 (헤드셋 영상으로 팔을 움직이기 전, 한 번)</div>${directionHtml(s)}
    <div class="actions">${logButton("quest_view")}</div>`;
}

function detailNeck(s) {
  const h = s.head;
  const rows = h ? [["pan (home 기준)", `${fmt(h.meas_pan_deg - h.home_pan_deg)}°  [오른 ${fmt(h.window_deg?.[0], 0)} / 왼 ${fmt(h.window_deg?.[1], 0)}]`],
    ["tilt (home 기준)", `${fmt(h.meas_tilt_deg - h.home_tilt_deg)}°  [아래 ${fmt(h.window_deg?.[2], 0)} / 위 ${fmt(h.window_deg?.[3], 0)}]`],
    ["HMD", h.hmd_tracked ? "추적 중" : "놓침", h.hmd_tracked ? "ok" : "warn"]] : [["상태", running("head") ? "기록 기다림" : "꺼짐"]];
  const port = (s.probes || {}).head_port || {};
  rows.push(["포트", !port.exists ? "없음" : (port.holders || []).length ? `PID ${port.holders.join(",")} 사용` : "비어 있음"]);
  return `<h3>목 · 다이나믹셀</h3>${facts(rows)}<p class="hint">시작하면 home 으로 가서 잠긴 채 기다린다. Space 로 따라가기/잠금, 잠그면 home 으로 돌아간다.</p>
    <div class="actions">${logButton("head")}</div>`;
}

// direct (Nova2Dex, 10.11 user): the robot hand reads the SenseGlove SDK's 0..1 values, so the SDK's
// calibration of each glove is the only one; the older calibrations stay folded away.
function detailGloveDirect(s, g, rows, calib) {
  const sides = Object.keys(g.gloves || {});
  const recal = sides.map((side) => btn(`${SIDE_KO[side]}손 장갑 보정 다시`, { "data-action": `glove_recal:${side}` }, "small")).join("");
  return `<h3>장갑 · Nova 2</h3>${facts(rows)}
    <p class="hint"><b>보정은 하나뿐이다 (장갑 SDK 보정).</b> 장갑 드라이버를 켜면 손마다 시작되고 음성으로 안내한다:
      다섯 손가락을 끝까지 쥐었다 펴고 엄지를 크게 돌리면 "보정 완료". 장갑을 다시 켜거나 손이 어긋나 보이면
      [장갑 보정 다시] (로봇 손은 [홈] / [주먹] 으로 둔 채). 파일럿 이름이나 보정 파일은 필요 없다.</p>
    <div class="actions">${recal}
      ${btn("드라이버 정지", { "data-action": "glove_driver_stop", disabled: !(g.driver_pids || []).length }, "ghost small")}</div>
    <details class="legacy"><summary>이전 방식 보정 (지금 설정 method: direct 에서는 쓰지 않음)</summary>
      <p class="hint">configs/hands/nova2_to_rh56f1.yaml 의 method 를 flexion / examples 로 되돌릴 때만 쓴다.</p>
      <div class="actions">${btn("오른손 보정", { "data-start": "calib_right" }, "small")}${btn("왼손 보정", { "data-start": "calib_left" }, "small")}
        ${btn("오른손 편 손 맞춤", { "data-start": "rezero_right" }, "small")}${btn("왼손 편 손 맞춤", { "data-start": "rezero_left" }, "small")}
        ${btn("오른손 손가락별 맞춤", { "data-start": "fingers_right" }, "small")}${btn("왼손 손가락별 맞춤", { "data-start": "fingers_left" }, "small")}
        ${btn("오른손 손가락 튜닝", { "data-start": "tune_right" }, "small")}${btn("왼손 손가락 튜닝", { "data-start": "tune_left" }, "small")}
        ${btn("Enter: 이 자세 기록", { "data-key": calib || "", "data-text": "\n", disabled: !calib }, "key small")}</div></details>
    <div class="actions">${logButton(calib || "task_glove_up")}</div>`;
}

function detailGlove(s) {
  const g = (s.probes || {}).gloves || {};
  const ros = (s.probes || {}).ros || {};
  const rows = [["SenseCom", !s.sensecom_installed ? "미설치 (scripts/ros_ws_setup.sh --full)" : g.sensecom_started_at ? "실행 중" : "꺼짐",
    s.sensecom_installed ? (g.sensecom_started_at ? "ok" : "") : "bad"]];
  for (const [side, glove] of Object.entries(g.gloves || {})) {
    rows.push([`${SIDE_KO[side]}손 ${glove.serial}`, `${glove.connected ? "BLE 연결" : "BLE 미연결"} · 토픽 ${(ros.glove_topics || {})[side] ? "있음" : "없음"}`]);
    const cal = (s.calibration || {})[side];
    if (cal && cal.sdk) rows.push([`${SIDE_KO[side]}손 장갑 보정`, cal.detail.replace("장갑 SDK 보정: ", ""), cal.ok ? "ok" : "warn"]);
    else if (cal) rows.push([`${SIDE_KO[side]}손 보정`, cal.ok && !cal.stale ? `유효 · 손가락별 맞춤 ${cal.sweeps ? "있음" : "없음"}` : cal.detail, cal.ok && !cal.stale ? "ok" : "warn"]);
  }
  rows.push(["드라이버", (g.driver_pids || []).length ? `PID ${g.driver_pids.join(", ")}` : "꺼짐"]);
  if (jobLine("glove")) rows.push(["최근", jobLine("glove"), job("glove").state === "failed" ? "bad" : ""]);
  const calib = CALIB_KEYS.find(running);
  if (s.hand_method === "direct" && s.mode === "real") return detailGloveDirect(s, g, rows, calib);
  return `<h3>장갑 · Nova 2</h3>${facts(rows)}
    <p class="hint">파일럿: <b>${esc(s.settings.user || "없음 (맨 위 오른쪽에 이름을 넣는다)")}</b>. 처음 한 번: [오른손 보정] → 화면 안내 예시 자세(11개)마다 [Enter: 이 자세 기록] → [왼손 보정]. 보정은 파일럿마다 저장되어 다음부터 그대로 쓴다. SenseCom 을 다시 켠 뒤 손이 어긋나면 [편 손 맞춤]: 손을 펴고 Enter 한 번(2초). 한 손가락을 움직일 때 다른 로봇 손가락이 따라 움직이면 [손가락별 맞춤]: 엄지·검지·중지·약지·주먹을 하나씩, 편 손 1초 뒤 그 손가락만 6초 굽혔다 펴기(보정 뒤에, 로봇 손 노드는 따라가는 중이 아닐 때 새 값을 읽는다). [손가락 튜닝]: 창이 뜬다. 관절마다 로봇 각도(로봇이 움직일 수 있는 범위 안)를 정하고 장갑 손가락을 그 자세로 둔 채 [지금 자세로 기록] → [저장] → 로봇 손 [홈] → [따라가기].</p>
    <div class="actions">${btn("오른손 보정", { "data-start": "calib_right" }, "small")}${btn("왼손 보정", { "data-start": "calib_left" }, "small")}
      ${btn("오른손 편 손 맞춤", { "data-start": "rezero_right" }, "small")}${btn("왼손 편 손 맞춤", { "data-start": "rezero_left" }, "small")}
      ${btn("오른손 손가락별 맞춤", { "data-start": "fingers_right" }, "small")}${btn("왼손 손가락별 맞춤", { "data-start": "fingers_left" }, "small")}
      ${btn("오른손 손가락 튜닝", { "data-start": "tune_right" }, "small")}${btn("왼손 손가락 튜닝", { "data-start": "tune_left" }, "small")}
      ${btn("Enter: 이 자세 기록", { "data-key": calib || "", "data-text": "\n", disabled: !calib }, "key small")}
      ${btn("드라이버 정지", { "data-action": "glove_driver_stop", disabled: !(g.driver_pids || []).length }, "ghost small")}</div>
    <div class="actions">${logButton(calib || "task_glove_up")}</div>`;
}

function detailArm(s) {
  const p = s.probes || {};
  const rows = Object.entries(s.station.can_ports).map(([side, port]) => {
    const c = (p.can || {})[side] || {};
    return [`CAN ${SIDE_KO[side]}팔`, `${port} ${c.up ? "UP" : "DOWN"}${c.fd ? " FD" : ""}`, c.up && c.fd ? "ok" : real() ? "bad" : ""];
  });
  const holders = Array.isArray(p.can_holders) ? p.can_holders : [];
  rows.push(["s2r CAN 점유", holders.length ? holders[0].slice(0, 60) : "없음", holders.length ? "bad" : "ok"]);
  rows.push(["TCP 보정", s.station.tcp.measured ? s.station.tcp.path : "identity (측정 전)"]);
  rows.push(["RT 한도", p.rt ? `rtprio ${p.rt.rtprio}` : "-", p.rt?.ok ? "ok" : "warn"]);
  return `<h3>로봇 팔 · OpenArm</h3>${facts(rows)}
    <div class="row"><label class="field">이동 배율 <input type="number" id="scale" min="0.1" max="1.7" step="0.1" value="${esc(s.settings.scale)}" ${["arm", "record", "arm_right", "arm_left"].some(running) ? "disabled" : ""}></label></div>
    <p class="hint">한 팔씩: 팔 칸의 [홈] = 차렷 → 저장 경로 → home (장갑 없이, 손은 주먹 그대로), [따라가기] = Space,
      따라가는 중 [홈] = home 으로 돌아가 다시 기다림, [차렷] = home → 차렷 → 모터 끔. 돌아가다 실패하면 팔은 그 자리에서 버티고
      [다시 홈] / [차렷] / 받치고 [모터 끄기] 를 고른다. 손 칸의 [홈]/[따라가기]/[주먹]은 팔과 따로 움직인다.
      모두 확인되면 가운데 [전체 시작]: 양팔 + 양손, Space 하나로 같이 따라간다.</p>
    <div class="actions">${logButton("arm_right").replace("전체 로그", "오른팔 로그")}${logButton("arm_left").replace("전체 로그", "왼팔 로그")}${logButton("arm").replace("전체 로그", "전체 시작 로그")}</div>`;
}

function detailHand(s, side) {
  const key = `hand_${side}`;
  const ros = (s.probes || {}).ros || {};
  const h = (ros.hands || {})[side] || {};
  const rec = (s.hands || {})[side];
  const rows = [["ROS 도메인", ros.error ? `오류: ${ros.error}` : ros.domain ?? "-"],
    ["EtherCAT 드라이버", driverText(s, side), h.driver ? "ok" : ""],
    ["angle_set 발행", (h.angle_set_publisher_nodes || []).join(", ") || String(h.angle_set_publishers ?? "-")]];
  if (rec) {
    rows.push(["상태", `${rec.mode} · ${rec.phase || "-"} · ${rec.state} · ${rec.manual ? "수동 " + rec.manual : "자동"}`, rec.fault ? "bad" : ""]);
    rows.push(["장갑 나이", `${fmt(rec.glove_age_s, 2)} s${rec.glove_frozen ? " (멈춤)" : ""}`, rec.glove_frozen ? "warn" : ""]);
    if (rec.refusal) rows.push(["켜기 거부", rec.refusal, "bad"]);
  }
  const cal = (s.calibration || {})[side];
  if (cal) rows.push(["장갑 보정", cal.detail, cal.ok && !cal.stale ? "ok" : "warn"]);
  if (jobLine(key)) rows.push(["최근", jobLine(key), job(key).state === "failed" ? "bad" : ""]);
  const driverOff = real() && running(`ecat_${side}`) && !running(key)
    ? btn("드라이버 내리기", { "data-action": `driver_off:${side}` }, "ghost small") : "";
  return `<h3>로봇 ${SIDE_KO[side]}손 · RH56F1</h3>${facts(rows)}
    <p class="hint">[홈]: 드라이버·노드(없으면) → home(펼침)에서 멈춤. [따라가기]: home → 1초 기준 자세(로봇 손 home 과 같은 모양: 엄지·검지 C자) → 장갑을 따라감, 팔이 home 에 있어도. 둘 다 FAULT 도 풀고 다시 켠다. [끄기]: 주먹으로 쥔 뒤 노드를 내린다(드라이버는 남는다). 팔이 경로를 도는 동안은 언제나 주먹. 전체 시작은 손을 자동(팔이 따라가면 같이)으로 되돌린다. 장갑 보정이 먼저 있어야 한다.</p>
    <div class="actions">${logButton(key)}${real() ? logButton(`ecat_${side}`).replace("전체 로그", "드라이버 로그") : ""}${driverOff}</div>`;
}

function detailRecord(s) {
  const rows = (s.record_checklist || []).map((i) => [i.name, i.detail, i.ok ? "ok" : "warn"]);
  return `<h3>녹화</h3>${facts(rows)}
    <div class="row"><label class="field grow">작업 설명 <input type="text" id="task" maxlength="200" value="${esc(s.settings.task)}"></label>
      <label class="field">에피소드 <input type="number" id="episodes" min="1" max="500" step="1" value="${esc(s.settings.episodes)}"></label></div>
    <p class="hint">시작하면 팔이 차렷 → home 으로 간다. Space 로 에피소드 시작/저장, R 은 버리고 다시, Q 는 마침. 빠진 스트림(! 표시)은 데이터셋에 안 들어간다.</p>
    <div class="actions">${logButton("record")}</div>`;
}

const DETAILS = {
  quest: detailQuest, neck: detailNeck, glove_left: detailGlove, glove_right: detailGlove,
  arm_left: detailArm, arm_right: detailArm, hand_left: (s) => detailHand(s, "left"), hand_right: (s) => detailHand(s, "right"),
  record: detailRecord,
};
const DETAIL_TAIL = { quest: "quest_view", neck: "head", arm_left: "arm_left", arm_right: "arm_right", hand_left: "hand_left",
  hand_right: "hand_right", record: "record", glove_left: "calib", glove_right: "calib" };

function renderDetail(s) {
  const box = $("#detail");
  const html = DETAILS[selected](s);
  let tailKey = DETAIL_TAIL[selected];
  if (tailKey === "calib") tailKey = CALIB_KEYS.find(running) || "task_glove_up";
  if (/^arm_/.test(tailKey) && !running(tailKey) && (running("arm") || unit(tailKey).rc == null)) tailKey = "arm";
  if (box.dataset.html !== html && !box.contains(document.activeElement)) {
    box.innerHTML = html + `<pre class="tail" id="detail-tail"></pre>`;
    box.dataset.html = html;
  }
  const u = unit(tailKey);
  const lines = (u.tail || []).slice(-10);
  if (u.partial) lines.push(u.partial);
  const pre = $("#detail-tail");
  if (pre && pre.textContent !== lines.join("\n")) { pre.textContent = lines.join("\n"); pre.scrollTop = pre.scrollHeight; }
}

// ---------------------------------------------------------------- top, alerts, bar
function renderTop(s) {
  $("#station").textContent = `${s.station.name} · ${s.station.robot}`;
  const pilot = $("#user");
  if (document.activeElement !== pilot && pilot.value !== (s.settings.user || "")) pilot.value = s.settings.user || "";
  document.body.classList.toggle("is-real", s.mode === "real");
  const busy = s.running.length > 0;
  $$("#top [data-mode]").forEach((b) => {
    b.setAttribute("aria-checked", String(b.dataset.mode === s.mode));
    b.disabled = busy && b.dataset.mode !== s.mode;
    b.title = b.disabled ? "실행 중인 것을 모두 정지한 뒤 바꿀 수 있다" : "";
  });
  const p = s.probes || {};
  const holders = Array.isArray(p.can_holders) ? p.can_holders : null;
  const canOk = Object.keys(s.station.can_ports).every((side) => (p.can || {})[side]?.up && (p.can || {})[side]?.fd);
  const pills = [["CAN", canOk ? "UP" : "DOWN", canOk ? "ok" : real() ? "bad" : "mute"],
    ["s2r CAN", holders === null ? "?" : holders.length ? "점유" : "비어 있음", holders && !holders.length ? "ok" : "bad"],
    ["RT", p.rt ? String(p.rt.rtprio) : "-", p.rt?.ok ? "ok" : "warn"]];
  if (s.station.hands.length) {
    const ros = p.ros || {};
    const drivers = s.station.hands.filter((side) => (ros.hands || {})[side]?.driver).length;
    pills.push([`손 드라이버 · 도메인 ${ros.domain ?? "-"}`, ros.error ? "오류" : `${drivers}/${s.station.hands.length}`, drivers ? "ok" : "warn"]);
  }
  const html = pills.map(([k, v, t]) => `<span class="badge ${t}">${esc(k)} ${esc(v)}</span>`).join("");
  if ($("#health").dataset.html !== html) { $("#health").innerHTML = html; $("#health").dataset.html = html; }
  document.title = `macq 콘솔 · ${s.station.name} · ${s.mode === "real" ? "실기" : "FAKE"}`;
}

function renderAlerts(s) {
  const items = [];
  if (s.closing)
    items.push(`<div class="alert warn"><p><b>콘솔 종료 중</b>: 모든 것을 안전 자세로 정지하는 중이다. 끝나면 서버가 내려간다(새 시작 불가).</p></div>`);
  for (const e of s.left_behind || [])
    items.push(`<div class="alert"><p>이전 콘솔이 남긴 <b>${esc(unitName(e.key))}</b> (PID ${Number(e.pid)}) 가 아직 돈다${e.stopping ? " (정지 중)" : ""}: <code>${esc((e.argv || []).join(" ").slice(0, 160))}</code></p>
      <button type="button" class="btn danger" data-left="${Number(e.pid)}" ${e.stopping ? "disabled" : ""}>정지 (SIGINT)</button></div>`);
  for (const [key, u] of Object.entries(s.units || {})) {
    if (!u.prompt) continue;
    const yn = /\[y\/n\]/i.test(u.prompt);
    if (recovering(key)) {
      items.push(`<div class="alert"><p><b>${esc(unitName(key))}</b> 이 멈춰서 버티는 중: <code>${esc(u.prompt)}</code></p>
        <div class="actions">${recoveryButtons(key)}</div></div>`);
      continue;
    }
    items.push(`<div class="alert ${CALIB_KEYS.includes(key) ? "warn" : ""}"><p><b>${esc(unitName(key))}</b> 이 입력을 기다린다: <code>${esc(u.prompt)}</code></p>
      ${yn ? `<button type="button" class="btn" data-send="${esc(key)}" data-text="y&#10;">y</button><button type="button" class="btn" data-send="${esc(key)}" data-text="n&#10;">n</button>`
           : `<button type="button" class="btn danger" data-send="${esc(key)}" data-text="&#10;">Enter 보내기</button>`}</div>`);
  }
  const html = items.join("");
  if ($("#alerts").dataset.html !== html) { $("#alerts").innerHTML = html; $("#alerts").dataset.html = html; }
}

function renderBar(s) {
  const chips = s.running.map((k) => `<button type="button" class="chip ${stopping(k) ? "stopping" : ""}" data-log="${esc(k)}">${esc(unitName(k))}</button>`).join("");
  if ($("#running").dataset.html !== chips) { $("#running").innerHTML = chips; $("#running").dataset.html = chips; }
  const sel = $("#space-target");
  if (document.activeElement !== sel) sel.value = s.space_target || "";
  $$("[data-key]").forEach((b) => b.classList.toggle("target", b.dataset.key === s.space_target && b.dataset.text === " "));
  $("#stop-all").disabled = s.running.length === 0;
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
  if (!DETAILS[selected] || nodeHidden(s, selected)) selected = "quest";
  renderTop(s);
  renderAlerts(s);
  renderNodes(s);
  renderDetail(s);
  renderBar(s);
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
  const node = ev.target.closest(".node");
  if (!b) {
    if (node && !ev.target.closest("label, input")) { selected = node.dataset.node; if (S) render(S); }
    return;
  }
  const d = b.dataset;
  if (d.mode && !b.disabled && S && d.mode !== S.mode) {
    if (d.mode === "real" && !(await confirmDialog("실기 모드로", "이제부터 시작하는 것은 실제 로봇·장치를 쓴다. 움직이는 것은 시작 전에 명령을 다시 보여 준다.", "", true))) return;
    act("mode", { mode: d.mode }, null, d.mode === "real" ? "실기 모드" : "FAKE 모드");
  } else if (d.start) act("start", { key: d.start, opts: d.opts ? JSON.parse(d.opts) : undefined }, b, `${unitName(d.start)} 시작`);
  else if (d.stop) act("stop", { key: d.stop }, b, `${unitName(d.stop)} 정지 요청 (안전 자세로)`);
  else if (d.key !== undefined && d.text !== undefined && d.key) act("key", { key: d.key, text: d.text }, null, `${unitName(d.key)} ← ${d.text === " " ? "Space" : d.text === "h" ? "홈(H)" : d.text.trim() || "Enter"}`);
  else if (d.send) {
    if (d.confirm && !(await confirmDialog("모터 끄기", d.confirm, "", true))) return;
    act("key", { key: d.send, text: d.text }, b, "입력 보냄");
  }
  else if (d.action) act("action", { name: d.action }, b, "요청함");
  else if (d.log) openDrawer(d.log);
  else if (d.select) { selected = d.select; if (S) render(S); if (!S?.settings.user) setTimeout(() => $("#user")?.focus(), 50); }
  else if (d.left) act("action", { name: `left_behind_stop:${d.left}` }, b, "SIGINT 보냄");
});

document.addEventListener("change", (ev) => {
  const t = ev.target;
  const field = { scale: ["scale", Number], task: ["task", String], episodes: ["episodes", Number], user: ["user", String] }[t.id];
  if (field) act("settings", { [field[0]]: field[1](t.value) }, null, "설정 저장");
});

$("#space-target").addEventListener("change", (ev) => act("space_target", { key: ev.target.value || null }, null, "Space 대상 바꿈"));
$("#stop-all").addEventListener("click", async (ev) => {
  if (await confirmDialog("모두 정지", "실행 중인 모든 것에 SIGINT 를 한 번 보낸다. 팔·녹화는 home → 차렷, 목은 home. 로봇 손은 팔이 차렷에 간 뒤 펼침으로, 손 드라이버는 마지막에 내린다.", "", false))
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
  if ((ev.key === "Enter" || ev.key === " ") && ev.target.classList?.contains("node")) {  // keyboard select
    ev.preventDefault(); selected = ev.target.dataset.node; if (S) render(S); return;
  }
  if (ev.code !== "Space" || ev.repeat || $("#confirm").open) return;
  if (ev.target.closest("input, textarea, select")) return;
  ev.preventDefault();  // Space goes to the robot program, not to the focused button
  if (!S?.space_target) { toast("Space 대상 없음: 아래 막대에서 고른다", "bad"); return; }
  act("key", { key: S.space_target, text: " " }, null, `${unitName(S.space_target)} ← Space`);
});

// ---------------------------------------------------------------- start
(function init() {
  const params = new URLSearchParams(location.search);
  const saved = params.get("theme") || localStorage.getItem("macq-theme");
  document.documentElement.dataset.theme = saved || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  if (params.get("node")) selected = params.get("node");
  if (params.has("once")) {  // one render, no stream (headless screenshots)
    fetch("/api/state").then((r) => r.json()).then((s) => { setBadge(["ok", "한 번 읽음"]); render(s); });
    return;
  }
  const es = new EventSource("/api/stream");
  es.addEventListener("state", (ev) => { setBadge(["ok", "연결됨"]); render(JSON.parse(ev.data)); });
  es.onerror = () => setBadge(["bad", "콘솔 서버 끊김"]);
  setInterval(refreshDrawer, 1000);
})();
