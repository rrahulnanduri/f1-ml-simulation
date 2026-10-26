// A very small Chrome DevTools driver, enough to screenshot and script the built pages.
// Edge's old headless mode is used because the new mode refuses to start in some locked-down setups.
// Everything it starts is tracked by its private profile folder and stopped by that, never by name.
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";

const CANDIDATES = [
  process.env.CHROME_PATH,
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].filter(Boolean);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function launch({ width = 1200, height = 800, scale = 1.5, headless = process.env.CAPTURE_HEADLESS || "old" } = {}) {
  const binary = CANDIDATES.find((p) => existsSync(p));
  if (!binary) throw new Error("No Chrome or Edge found. Set CHROME_PATH.");
  const profile = mkdtempSync(join(tmpdir(), "f1a-cdp-"));
  const port = 9300 + Math.floor(Math.random() * 500);
  spawn(binary, [`--headless=${headless}`, `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "--no-first-run", "--no-default-browser-check", "--hide-scrollbars", "--allow-file-access-from-files", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--disable-background-timer-throttling", "--disable-renderer-backgrounding", `--window-size=${width},${height}`, "about:blank"], { stdio: "ignore", detached: true }).unref();

  let version = null;
  for (let i = 0; i < 80 && !version; i++) {
    try {
      version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
    } catch {
      await sleep(250);
    }
  }
  if (!version) {
    cleanup(profile);
    throw new Error("browser did not open a DevTools port");
  }
  const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })).json();
  const socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((ok, bad) => {
    socket.onopen = ok;
    socket.onerror = bad;
  });
  let id = 0;
  const pending = new Map();
  const logs = [];
  const listeners = new Map();
  socket.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) {
      const { ok, bad } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? bad(new Error(msg.error.message)) : ok(msg.result);
    } else if (msg.method) {
      if (msg.method === "Runtime.consoleAPICalled" && ["error", "warning", "assert"].includes(msg.params.type)) logs.push(`${msg.params.type}: ${msg.params.args.map((a) => a.value ?? a.description).join(" ")}`);
      if (msg.method === "Runtime.exceptionThrown") logs.push("exception: " + (msg.params.exceptionDetails.exception?.description || msg.params.exceptionDetails.text));
      (listeners.get(msg.method) || []).forEach((fn) => fn(msg.params));
    }
  };
  const send = (method, params = {}) =>
    new Promise((ok, bad) => {
      const mine = ++id;
      pending.set(mine, { ok, bad });
      socket.send(JSON.stringify({ id: mine, method, params }));
    });
  const once = (method) => new Promise((ok) => { const list = listeners.get(method) || []; list.push((p) => ok(p)); listeners.set(method, list); });

  await send("Page.enable");
  await send("Runtime.enable");
  let current = { width, height, scale };
  const metrics = (w, h, s = current.scale) => {
    current = { width: w, height: h, scale: s };
    return send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: s, mobile: false });
  };
  await metrics(width, height, scale);

  const api = {
    logs,
    send,
    metrics,
    async open(url, settle = 800) {
      listeners.delete("Page.loadEventFired");
      const loaded = once("Page.loadEventFired");
      await send("Page.navigate", { url });
      await Promise.race([loaded, sleep(20000)]);
      await sleep(settle);
    },
    async eval(expression) {
      const r = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
      if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
      return r.result.value;
    },
    sleep,
    /** Screenshot of the viewport, or of an element (CSS selector), or of a clip in CSS px. */
    async shot(file, { selector, clip, scale: s } = {}) {
      let region = clip;
      if (selector) {
        region = await api.eval(`(() => { const r = document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect(); return { x: r.left + scrollX, y: r.top + scrollY, width: r.width, height: r.height }; })()`);
      }
      const params = { format: "png", captureBeyondViewport: true };
      if (region) params.clip = { x: region.x, y: region.y, width: region.width, height: region.height, scale: s || 1 };
      const r = await send("Page.captureScreenshot", params);
      writeFileSync(resolve(file), Buffer.from(r.data, "base64"));
      return file;
    },
    async key(key) {
      const code = key.length === 1 ? { text: key } : {};
      await send("Input.dispatchKeyEvent", { type: "keyDown", key, ...code });
      await send("Input.dispatchKeyEvent", { type: "keyUp", key });
    },
    async mouse(type, x, y, extra = {}) {
      await send("Input.dispatchMouseEvent", { type, x, y, button: "left", buttons: type === "mouseReleased" ? 0 : 1, clickCount: 1, ...extra });
    },
    async close() {
      try { await send("Browser.close"); } catch {}
      try { socket.close(); } catch {}
      await sleep(300);
      cleanup(profile);
    },
  };
  return api;
}

function cleanup(profile) {
  const name = basename(profile);
  if (process.platform === "win32") {
    spawnSync("powershell", ["-NoProfile", "-Command", `Get-CimInstance Win32_Process -Filter "Name='msedge.exe' OR Name='chrome.exe'" | Where-Object { $_.CommandLine -like '*${name}*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }`], { stdio: "ignore" });
  } else {
    spawnSync("pkill", ["-f", name], { stdio: "ignore" });
  }
  try { rmSync(profile, { recursive: true, force: true }); } catch {}
}

export const pageUrl = (root, page, hash = "") => "file:///" + resolve(root, "site", `${page}.html`).replace(/\\/g, "/") + (hash ? "#" + hash : "");
