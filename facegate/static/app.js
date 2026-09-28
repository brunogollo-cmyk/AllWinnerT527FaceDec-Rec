"use strict";

// --- tiny helpers ---------------------------------------------------------

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function toast(message, isError) {
  const el = $("#toast");
  el.textContent = message;
  el.className = isError ? "err" : "";
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, 3500);
}

async function api(path, options) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch (_) { /* empty body */ }
  if (!res.ok) throw new Error((body && body.error) || res.statusText);
  return body;
}

const post = (path, payload) => api(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload || {}),
});

const fmt = (n, d = 2) =>
  n === null || n === undefined ? "—" : Number(n).toFixed(d);

// --- tabs -----------------------------------------------------------------

$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.remove("active"));
    $$(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $("#panel-" + tab.dataset.tab).classList.add("active");
    if (tab.dataset.tab === "people") loadPeople();
    if (tab.dataset.tab === "settings") loadSettings();
    // Only pull frames while the tab is actually on screen. An <img> pointed at
    // /stream keeps a server-side connection open even when scrolled away, so
    // leaving it on would cost a second MJPEG consumer for nothing.
    setPreviewStream(tab.dataset.tab === "people");
  });
});

// The People tab shows the camera so the operator can frame themselves before
// capturing. It shares the /stream endpoint with the Live tab rather than
// opening the camera twice - the slot serves both consumers from the same JPEG.
let previewOn = false;
function setPreviewStream(on) {
  if (on === previewOn) return;
  previewOn = on;
  const img = $("#previewStream");
  if (on) {
    img.src = "/stream?preview=" + Date.now();
  } else {
    img.removeAttribute("src");
  }
}

// --- status ---------------------------------------------------------------

let streaming = true;

async function loadStatus() {
  try {
    const s = await api("/api/status");
    const parts = [
      `<b>${fmt(s.fps, 1)}</b> fps`,
      `${fmt(s.detect_ms, 1)}ms detect`,
      `${fmt(s.embed_ms, 1)}ms embed`,
    ];
    if (s.liveness_enabled) parts.push(`${fmt(s.liveness_ms, 1)}ms liveness`);
    parts.push(`${s.people} enrolled`);
    parts.push(s.camera || "no camera");
    if (!s.camera_connected) parts.push("<b>disconnected</b>");
    $("#statline").innerHTML = parts.join(" · ");
  } catch (e) {
    $("#statline").textContent = "offline";
  }
}

// --- live -----------------------------------------------------------------

$("#toggleStream").addEventListener("click", () => {
  streaming = !streaming;
  $("#stream").style.display = streaming ? "" : "none";
  $("#videoOff").hidden = streaming;
  $("#toggleStream").textContent = streaming ? "Pause view" : "Resume";
});

$("#resume").addEventListener("click", () => $("#toggleStream").click());

$("#snap").addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = "/api/snapshot";
  a.download = "facegate-" + Date.now() + ".jpg";
  a.click();
});

// Reconnect the stream if the service restarts.
setInterval(() => {
  const img = $("#stream");
  if (streaming && img.naturalWidth === 0 && img.complete) {
    img.src = "/stream?t=" + Date.now();
  }
}, 4000);

// --- people ---------------------------------------------------------------

// Captures live in the browser: each holds a thumbnail and the 128-d embedding
// so enrolling does not need the server to keep per-capture state.
let captures = [];

async function loadPeople() {
  try {
    const data = await api("/api/people");
    const tbody = $("#peopleTable tbody");
    tbody.innerHTML = "";
    $("#peopleEmpty").hidden = data.people.length > 0;
    $("#peopleTable").hidden = data.people.length === 0;

    data.people.forEach((p) => {
      const tr = document.createElement("tr");

      const name = document.createElement("td");
      name.textContent = p.name;
      tr.appendChild(name);

      const count = document.createElement("td");
      count.className = "num";
      count.textContent = p.embeddings;
      tr.appendChild(count);

      const sim = document.createElement("td");
      sim.className = "num sim";
      if (p.similarity === null) {
        sim.textContent = "n/a";
        sim.classList.add("mid");
      } else {
        sim.textContent = fmt(p.similarity, 3);
        sim.classList.add(
          p.similarity >= 0.8 ? "good" : p.similarity >= 0.7 ? "mid" : "bad"
        );
      }
      tr.appendChild(sim);

      const actions = document.createElement("td");
      actions.className = "num";
      const btn = document.createElement("button");
      btn.className = "secondary danger";
      btn.textContent = "Remove";
      btn.addEventListener("click", () => removePerson(p.name, btn));
      actions.appendChild(btn);
      tr.appendChild(actions);

      tbody.appendChild(tr);
    });
  } catch (e) {
    toast("Could not load people: " + e.message, true);
  }
}

async function removePerson(name, btn) {
  if (!confirm(`Remove ${name}? Their face data will be deleted.`)) return;
  btn.disabled = true;
  try {
    await post("/api/people/remove", { name });
    toast(`Removed ${name}`);
    loadPeople();
    loadStatus();
  } catch (e) {
    toast(e.message, true);
    btn.disabled = false;
  }
}

$("#captureBtn").addEventListener("click", async () => {
  const btn = $("#captureBtn");
  btn.disabled = true;
  btn.textContent = "Capturing…";
  try {
    const r = await post("/api/capture");
    if (!r.ok) {
      toast("No face captured: " + (r.reason || "unknown"), true);
      // The preview is there to fix exactly this, so say what to change
      // rather than leaving the operator guessing.
      setCaptureHint(
        r.reason === "no face found"
          ? "No face detected in the preview. Move into frame, face the camera, and improve the lighting."
          : "The face could not be aligned. Face the camera more squarely.",
        true
      );
      return;
    }
    // The server returns the 128-d embedding so the chosen set can be posted
    // back on enrol without any per-capture state on the server.
    captures.push({
      thumbnail: r.thumbnail,
      feature: r.feature,
      similarity: r.similarity,
      known: r.known,
      liveness: r.liveness,
    });
    renderCaptures();
    const n = captures.length;
    setCaptureHint(
      `Photo ${n} captured. Vary your angle or expression for the next one.`,
      false
    );
  } catch (e) {
    toast("Capture failed: " + e.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Capture photo";
  }
});

function setCaptureHint(text, isError) {
  const el = $("#captureHint");
  el.textContent = text;
  el.className = isError ? "hint err" : "hint";
}

$("#clearCaptures").addEventListener("click", () => {
  captures = [];
  renderCaptures();
});

function renderCaptures() {
  const wrap = $("#captures");
  wrap.innerHTML = "";
  $("#clearCaptures").hidden = captures.length === 0;
  $("#enrollBtn").disabled = captures.length === 0;

  captures.forEach((c, i) => {
    const box = document.createElement("div");
    box.className = "capture";

    const img = document.createElement("img");
    img.src = "data:image/jpeg;base64," + c.thumbnail;
    img.alt = "capture " + (i + 1);
    box.appendChild(img);

    const meta = document.createElement("div");
    meta.className = "meta";

    const label = document.createElement("span");
    if (c.known) {
      label.textContent = "matches " + c.known;
    } else {
      label.textContent = "no match " + fmt(c.similarity, 2);
    }
    meta.appendChild(label);

    const drop = document.createElement("button");
    drop.className = "drop";
    drop.textContent = "×";
    drop.title = "Discard this photo";
    drop.addEventListener("click", () => {
      captures.splice(i, 1);
      renderCaptures();
    });
    meta.appendChild(drop);

    box.appendChild(meta);
    wrap.appendChild(box);
  });
}

$("#enrollBtn").addEventListener("click", async () => {
  const name = $("#personName").value.trim();
  const msg = $("#enrollMsg");
  if (!name) {
    msg.textContent = "Enter a name first";
    msg.className = "msg err";
    return;
  }

  const btn = $("#enrollBtn");
  btn.disabled = true;
  btn.textContent = "Enrolling…";
  msg.textContent = "";
  msg.className = "msg";

  try {
    const r = await post("/api/enroll", {
      name,
      features: captures.map((c) => c.feature),
    });
    if (r.low_similarity) {
      toast(
        `${r.name} enrolled, but the photos are inconsistent ` +
          `(${fmt(r.similarity, 2)}). Capture more varied angles.`,
        true
      );
    } else {
      toast(`${r.name} enrolled from ${r.enrolled} photo(s)`);
    }
    captures = [];
    renderCaptures();
    $("#personName").value = "";
    loadPeople();
    loadStatus();
  } catch (e) {
    msg.textContent = e.message;
    msg.className = "msg err";
  } finally {
    btn.disabled = captures.length === 0;
    btn.textContent = "Enrol";
  }
});

// --- events ---------------------------------------------------------------

let eventTimer = null;

async function loadEvents() {
  try {
    const data = await api("/api/events?limit=100");
    const filter = $("#eventFilter").value;
    const rows = data.events.filter((e) => !filter || e.decision === filter);
    const tbody = $("#eventsTable tbody");
    tbody.innerHTML = "";
    $("#eventsEmpty").hidden = rows.length > 0;
    $("#eventsTable").hidden = rows.length === 0;

    rows.forEach((e) => {
      const tr = document.createElement("tr");

      const time = document.createElement("td");
      time.className = "time";
      time.textContent = (e.ts || "").replace("T", " ").slice(0, 19);
      tr.appendChild(time);

      const dec = document.createElement("td");
      const pill = document.createElement("span");
      pill.className =
        "pill " + (e.decision === "GRANT" ? "grant"
                 : e.decision === "DENY_SPOOF" ? "spoof" : "deny");
      pill.textContent = e.decision;
      dec.appendChild(pill);
      tr.appendChild(dec);

      const who = document.createElement("td");
      who.textContent = e.name || "—";
      tr.appendChild(who);

      const sim = document.createElement("td");
      sim.className = "num";
      sim.textContent = fmt(e.similarity, 2);
      tr.appendChild(sim);

      const live = document.createElement("td");
      live.className = "num";
      live.textContent = e.liveness === undefined ? "—" : fmt(e.liveness, 2);
      tr.appendChild(live);

      tbody.appendChild(tr);
    });
  } catch (e) {
    /* the service may be restarting; the next poll will pick it up */
  }
}

$("#eventFilter").addEventListener("change", loadEvents);

$("#clearEvents").addEventListener("click", async () => {
  if (!confirm("Delete the whole access log?")) return;
  try {
    await api("/api/events", { method: "DELETE" });
    toast("Access log cleared");
    loadEvents();
  } catch (e) {
    toast(e.message, true);
  }
});

// --- settings -------------------------------------------------------------

const CHECKBOXES = ["access.require_match", "access.require_liveness"];

function fillSettings(cfg) {
  CHECKBOXES.forEach((key) => {
    const [section, leaf] = key.split(".");
    const el = $("#" + leaf);
    if (el && section in cfg) el.checked = !!cfg[section][leaf];
  });

  const set = (id, value) => {
    const el = $("#" + id);
    if (el && value !== undefined) el.value = value;
  };
  set("liveness_threshold", cfg.access.liveness_threshold);
  set("match_threshold", cfg.recognizer.match_threshold);
  set("min_face_size", cfg.recognizer.min_face_size);
  set("event_cooldown", cfg.access.event_cooldown);
  set("input_size", cfg.detector.input_size);
  set("score_threshold", cfg.detector.score_threshold);
  set("camera_width", cfg.camera.width);
  set("camera_height", cfg.camera.height);
  set("camera_fps", cfg.camera.fps);
  set("camera_fourcc", cfg.camera.fourcc);
}

async function loadSettings() {
  try {
    fillSettings(await api("/api/config"));
  } catch (e) {
    toast("Could not load settings: " + e.message, true);
  }
}

$("#settingsForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const msg = $("#settingsMsg");
  const changes = {};
  CHECKBOXES.forEach((key) => {
    changes[key] = $("#" + key.split(".")[1]).checked;
  });
  [
    "liveness_threshold", "match_threshold", "min_face_size", "event_cooldown",
    "input_size", "score_threshold", "camera_width", "camera_height", "camera_fps",
  ].forEach((id) => { changes[id] = $("#" + id).value; });
  changes["camera.fourcc"] = $("#camera_fourcc").value;

  try {
    const r = await post("/api/config", changes);
    msg.className = "msg ok";
    if (r.rejected && r.rejected.length) {
      msg.textContent =
        "Rejected: " + r.rejected.map((x) => x.key + " (" + x.error + ")").join(", ");
    } else {
      msg.textContent = "Saved";
    }
    if (r.deferred_until_restart && r.deferred_until_restart.length) {
      msg.textContent += " — needs restart: " + r.deferred_until_restart.join(", ");
    }
    setTimeout(() => { msg.textContent = ""; }, 6000);
    loadStatus();
  } catch (e) {
    msg.className = "msg err";
    msg.textContent = e.message;
  }
});

// --- boot -----------------------------------------------------------------

loadStatus();
loadPeople();
loadEvents();
setInterval(loadStatus, 2000);
setInterval(loadEvents, 3000);
