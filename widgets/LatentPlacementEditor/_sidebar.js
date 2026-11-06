// Position preset sidebar for LatentPlacementEditor.

const POSITION_PRESETS = [
  { label: "↖", title: "Top Left",      pos: "top-left"      },
  { label: "↑", title: "Top Center",    pos: "top-center"    },
  { label: "↗", title: "Top Right",     pos: "top-right"     },
  { label: "←", title: "Left Center",   pos: "left-center"   },
  { label: "⊕", title: "Center",        pos: "center"        },
  { label: "→", title: "Right Center",  pos: "right-center"  },
  { label: "↙", title: "Bottom Left",   pos: "bottom-left"   },
  { label: "↓", title: "Bottom Center", pos: "bottom-center" },
  { label: "↘", title: "Bottom Right",  pos: "bottom-right"  },
];

// Offsets for a named position, snapped to the latent grid and kept inside the destination.
function calcPosition(posName, destW, destH, srcW, srcH, step) {
  const maxX = Math.max(0, destW - srcW);
  const maxY = Math.max(0, destH - srcH);
  const snapDown = (v) => Math.floor(v / step) * step;
  const midX = snapDown(maxX / 2), midY = snapDown(maxY / 2);
  const endX = snapDown(maxX),     endY = snapDown(maxY);
  const positions = {
    "top-left":      { x: 0,    y: 0    },
    "top-center":    { x: midX, y: 0    },
    "top-right":     { x: endX, y: 0    },
    "left-center":   { x: 0,    y: midY },
    "center":        { x: midX, y: midY },
    "right-center":  { x: endX, y: midY },
    "bottom-left":   { x: 0,    y: endY },
    "bottom-center": { x: midX, y: endY },
    "bottom-right":  { x: endX, y: endY },
  };
  return positions[posName] || positions["top-left"];
}

export function createSidebar({ getSizes, step, onApply }) {
  const el = document.createElement("div");
  el.className = "nodrag nowheel";
  el.style.cssText = [
    "display:flex", "flex-direction:column", "gap:3px",
    "width:84px", "flex-shrink:0",
    "background:var(--muted)", "border-radius:6px", "padding:4px",
    "box-sizing:border-box",
  ].join(";");

  const allBtns = [];

  function makeBtn({ label, title, onClick }) {
    const btn = document.createElement("button");
    btn.title = title;
    btn.textContent = label;
    btn.style.cssText = [
      "width:100%", "padding:4px 2px", "border-radius:4px",
      "border:1px solid var(--border)",
      "background:var(--background)", "color:var(--foreground)", "cursor:pointer",
      "font-size:10px", "font-weight:500", "line-height:1.3",
      "white-space:nowrap", "box-sizing:border-box",
    ].join(";");
    btn.addEventListener("mouseenter", () => { if (!btn.disabled) btn.style.background = "var(--accent)"; });
    btn.addEventListener("mouseleave", () => { btn.style.background = "var(--background)"; });
    for (const ev of ["mousedown", "pointerdown"]) btn.addEventListener(ev, (e) => e.stopPropagation());
    btn.addEventListener("click", (e) => { e.stopPropagation(); if (!btn.disabled) onClick(); });
    allBtns.push(btn);
    return btn;
  }

  const lbl = document.createElement("div");
  lbl.textContent = "Pos";
  lbl.style.cssText = "font-size:9px;color:var(--muted-foreground);text-align:center;padding:2px 0;";
  el.appendChild(lbl);

  const posGrid = document.createElement("div");
  posGrid.style.cssText = "display:grid;grid-template-columns:repeat(3,1fr);gap:2px;";
  for (const { label, title, pos } of POSITION_PRESETS) {
    posGrid.appendChild(makeBtn({ label, title, onClick: () => {
      const { destW, destH, srcW, srcH } = getSizes();
      onApply(calcPosition(pos, destW, destH, srcW, srcH, step));
    }}));
  }
  el.appendChild(posGrid);

  function syncDisabled(off) {
    for (const btn of allBtns) {
      btn.disabled = off;
      btn.style.opacity = off ? "0.35" : "1";
      btn.style.cursor  = off ? "not-allowed" : "pointer";
    }
  }

  return { el, syncDisabled };
}
