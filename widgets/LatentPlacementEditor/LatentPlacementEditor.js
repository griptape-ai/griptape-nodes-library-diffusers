import { createSidebar } from './_sidebar.js';

const WIDGET_VERSION = "1.0.0";

// Offsets are divided by the VAE scale factor, so only multiples of it move the source.
const LATENT_STEP = 8;

const DEST_FILL     = "#262626";
const DEST_GRID     = "rgba(255,255,255,0.06)";
const DEST_BORDER   = "rgba(255,255,255,0.35)";
const SOURCE_FILL   = "rgba(27,176,244,0.25)";
const SOURCE_BORDER = "#1bb0f4";

export default function LatentPlacementEditor(container, props) {
  if (container._placementEditor?.wrapper?.isConnected) {
    container._placementEditor.handleUpdate(props);
    return { cleanup: container._placementEditor.cleanup, update: container._placementEditor.handleUpdate };
  }
  if (container._placementEditor) delete container._placementEditor;

  // ── State ──────────────────────────────────────────────────────────────────
  let latestValue = props.value || {};
  let onChangeRef = props.onChange;
  let isDisabled  = props.disabled || false;

  let destW = 0, destH = 0, srcW = 0, srcH = 0;
  let offX = 0, offY = 0;
  let resizeSource = false;
  let lockedParams = [];

  let scale = 1;
  let dragStart = null;

  // Sizes come only from a run and can be missing from a later value, so keep the last ones seen.
  function readValue(v) {
    if (v.dest_width > 0 && v.dest_height > 0 && v.source_width > 0 && v.source_height > 0) {
      destW = v.dest_width;  destH = v.dest_height;
      srcW  = v.source_width; srcH = v.source_height;
    }
    if (!dragStart) {
      offX = v.x_offset ?? 0;
      offY = v.y_offset ?? 0;
    }
    resizeSource = Boolean(v.resize_source);
    lockedParams = v.locked || [];
  }
  readValue(latestValue);

  function hasSizes() { return destW > 0 && destH > 0 && srcW > 0 && srcH > 0; }
  function placedW()  { return resizeSource ? destW : srcW; }
  function placedH()  { return resizeSource ? destH : srcH; }
  function xLocked()  { return lockedParams.includes("x_offset"); }
  function yLocked()  { return lockedParams.includes("y_offset"); }
  // Where the node actually pastes: it divides the offset by the latent step and drops the remainder.
  function landedX()  { return Math.floor(offX / LATENT_STEP) * LATENT_STEP; }
  function landedY()  { return Math.floor(offY / LATENT_STEP) * LATENT_STEP; }
  function canMove()  { return !isDisabled && hasSizes() && !resizeSource && !(xLocked() && yLocked()); }

  // ── DOM ────────────────────────────────────────────────────────────────────
  const wrapper = document.createElement("div");
  wrapper.className = "latent-placement-editor nodrag nowheel";
  wrapper.style.cssText = [
    "display:flex", "flex-direction:column", "gap:6px",
    "height:100%",
    "user-select:none", "-webkit-user-select:none",
  ].join(";");

  const canvasRow = document.createElement("div");
  canvasRow.style.cssText = [
    "display:flex", "flex-direction:row", "gap:6px",
    "flex:1 1 0", "min-height:180px",
  ].join(";");

  const canvasWrap = document.createElement("div");
  canvasWrap.style.cssText = [
    "position:relative", "background:#111", "border-radius:6px",
    "overflow:hidden", "flex:1 1 0", "min-height:0",
    "display:flex", "align-items:center", "justify-content:center",
  ].join(";");

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;cursor:default;";
  const ctx = canvas.getContext("2d");

  const sidebarInst = createSidebar({
    getSizes: () => ({ destW, destH, srcW: placedW(), srcH: placedH() }),
    step: LATENT_STEP,
    onApply({ x, y }) {
      if (!canMove()) return;
      if (!xLocked()) offX = x;
      if (!yLocked()) offY = y;
      emit();
      render();
    },
  });

  // ── Footer: resize toggle + status ─────────────────────────────────────────
  const controls = document.createElement("label");
  controls.className = "nodrag nowheel";
  controls.style.cssText = [
    "background:var(--muted)", "border-radius:4px", "padding:6px 8px",
    "display:flex", "align-items:center", "gap:8px",
    "font-size:11px", "color:var(--foreground)", "cursor:pointer",
  ].join(";");
  const resizeToggle = document.createElement("input");
  resizeToggle.type = "checkbox";
  for (const ev of ["mousedown", "pointerdown"]) resizeToggle.addEventListener(ev, (e) => e.stopPropagation());
  resizeToggle.addEventListener("change", (e) => {
    e.stopPropagation();
    if (isDisabled) return;
    resizeSource = resizeToggle.checked;
    emit();
    render();
  });
  controls.appendChild(resizeToggle);
  controls.append("Resize source to destination");

  const statusBar = document.createElement("div");
  statusBar.style.cssText = [
    "font-size:11px", "color:var(--muted-foreground)", "padding:4px 8px",
    "background:var(--muted)", "border-radius:4px",
    "display:flex", "justify-content:space-between", "align-items:center",
  ].join(";");
  const statusL = document.createElement("span");
  const statusR = document.createElement("span");
  statusR.style.cssText = "font-family:monospace;font-size:10px;color:var(--muted-foreground);opacity:0.6;";
  statusR.textContent = `v${WIDGET_VERSION}`;
  statusBar.appendChild(statusL);
  statusBar.appendChild(statusR);

  canvasWrap.appendChild(canvas);
  canvasRow.appendChild(canvasWrap);
  canvasRow.appendChild(sidebarInst.el);
  wrapper.appendChild(canvasRow);
  wrapper.appendChild(controls);
  wrapper.appendChild(statusBar);
  container.appendChild(wrapper);

  function syncUI() {
    sidebarInst.syncDisabled(!canMove());
    resizeToggle.checked = resizeSource;
    resizeToggle.disabled = isDisabled;
    controls.style.opacity = isDisabled ? "0.4" : "1";
  }

  // ── Emit ───────────────────────────────────────────────────────────────────
  function emit() {
    if (typeof onChangeRef === "function") {
      onChangeRef({
        ...latestValue,
        dest_width: destW, dest_height: destH, source_width: srcW, source_height: srcH,
        x_offset: offX, y_offset: offY, resize_source: resizeSource,
      });
    }
    syncUI();
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  function resizeCanvas() {
    if (!hasSizes()) {
      canvas.width = 400; canvas.height = 240;
      render();
      return;
    }
    const areaW = canvasWrap.clientWidth  || 480;
    const areaH = canvasWrap.clientHeight || 360;
    scale = Math.min(areaW / destW, areaH / destH);
    canvas.width  = Math.max(1, Math.round(destW * scale));
    canvas.height = Math.max(1, Math.round(destH * scale));
    render();
  }

  function updateStatus() {
    statusL.style.color = "";
    if (!hasSizes()) {
      statusL.textContent = "Run the node to show latent sizes";
      return;
    }
    const locked = ["x_offset", "y_offset"].filter(f => lockedParams.includes(f));
    if (locked.length > 0) {
      statusL.style.color = "#f59e0b";
      statusL.textContent = "\u{1F512} Connected: " + locked.join(", ");
      return;
    }
    statusL.textContent = `x: ${landedX()}  y: ${landedY()}  ·  ${placedW()} × ${placedH()} on ${destW} × ${destH} px`;
  }

  function render() {
    const cw = canvas.width, ch = canvas.height;
    ctx.clearRect(0, 0, cw, ch);

    if (!hasSizes()) {
      ctx.fillStyle = "#1a1a1a";
      ctx.fillRect(0, 0, cw, ch);
      ctx.fillStyle = "#555";
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Run the node to show placement", cw / 2, ch / 2);
      updateStatus();
      return;
    }

    // Destination canvas with a 64px grid
    ctx.fillStyle = DEST_FILL;
    ctx.fillRect(0, 0, cw, ch);
    ctx.strokeStyle = DEST_GRID;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let gx = 64; gx < destW; gx += 64) { ctx.moveTo(gx * scale, 0); ctx.lineTo(gx * scale, ch); }
    for (let gy = 64; gy < destH; gy += 64) { ctx.moveTo(0, gy * scale); ctx.lineTo(cw, gy * scale); }
    ctx.stroke();
    ctx.strokeStyle = DEST_BORDER;
    ctx.strokeRect(0.5, 0.5, cw - 1, ch - 1);

    // Source placement; the canvas clips whatever falls outside the destination
    const sx = landedX() * scale, sy = landedY() * scale, sw = placedW() * scale, sh = placedH() * scale;
    ctx.fillStyle = SOURCE_FILL;
    ctx.fillRect(sx, sy, sw, sh);
    ctx.strokeStyle = SOURCE_BORDER;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(sx, sy, sw, sh);

    ctx.fillStyle = "#fff";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(resizeSource ? "source (resized)" : "source", sx + sw / 2, sy + sh / 2);

    updateStatus();
  }

  // ── Pointer events ─────────────────────────────────────────────────────────
  function canvasPos(e) {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * (canvas.width / r.width), (e.clientY - r.top) * (canvas.height / r.height)];
  }

  function hitSource(px, py) {
    return px >= landedX() * scale && px <= (landedX() + placedW()) * scale
        && py >= landedY() * scale && py <= (landedY() + placedH()) * scale;
  }

  function clampAxis(v, dest, src) {
    const max = Math.floor(Math.max(0, dest - src) / LATENT_STEP) * LATENT_STEP;
    const snapped = Math.round(v / LATENT_STEP) * LATENT_STEP;
    return Math.max(0, Math.min(snapped, max));
  }

  canvas.addEventListener("pointerdown", (e) => {
    if (!canMove()) return;
    e.stopPropagation();
    const [px, py] = canvasPos(e);
    if (!hitSource(px, py)) return;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = "grabbing";
    dragStart = { px, py, x: landedX(), y: landedY(), rawX: offX, rawY: offY };
  });

  canvas.addEventListener("pointermove", (e) => {
    e.stopPropagation();
    const [px, py] = canvasPos(e);
    if (!dragStart) {
      canvas.style.cursor = canMove() && hitSource(px, py) ? "move" : "default";
      return;
    }
    if (!xLocked()) offX = clampAxis(dragStart.x + (px - dragStart.px) / scale, destW, srcW);
    if (!yLocked()) offY = clampAxis(dragStart.y + (py - dragStart.py) / scale, destH, srcH);
    render();
  });

  canvas.addEventListener("pointerup", (e) => {
    if (!dragStart) return;
    e.stopPropagation();
    const moved = (!xLocked() && offX !== dragStart.x) || (!yLocked() && offY !== dragStart.y);
    if (!moved) { offX = dragStart.rawX; offY = dragStart.rawY; }
    dragStart = null;
    canvas.style.cursor = "move";
    if (moved) emit();
    else render();
  });

  canvas.addEventListener("pointercancel", () => {
    if (!dragStart) return;
    offX = dragStart.rawX; offY = dragStart.rawY;
    dragStart = null;
    canvas.style.cursor = "default";
    render();
  });

  canvas.addEventListener("mousedown", (e) => e.stopPropagation());
  canvasWrap.addEventListener("pointerdown", (e) => e.stopPropagation());
  canvasWrap.addEventListener("mousedown",   (e) => e.stopPropagation());

  // ── ResizeObserver ─────────────────────────────────────────────────────────
  const ro = new ResizeObserver(() => resizeCanvas());
  ro.observe(canvasWrap);

  resizeCanvas();
  syncUI();

  // ── handleUpdate ───────────────────────────────────────────────────────────
  function handleUpdate(newProps) {
    onChangeRef = newProps.onChange;
    isDisabled  = newProps.disabled || false;
    latestValue = newProps.value || {};
    const prevW = destW, prevH = destH;
    readValue(latestValue);
    syncUI();
    if (destW !== prevW || destH !== prevH) resizeCanvas();
    else render();
  }

  // ── cleanup ────────────────────────────────────────────────────────────────
  function cleanup() {
    ro.disconnect();
    wrapper.remove();
    delete container._placementEditor;
  }

  container._placementEditor = { handleUpdate, cleanup, wrapper };
  return { cleanup, update: handleUpdate };
}
