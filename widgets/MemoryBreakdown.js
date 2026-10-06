const COMPONENT_COLOR_FAMILIES = {
  denoiser: ["#2a78d6", "#1c5cab", "#5598e7", "#104281"],
  text_encoder: ["#eb6834", "#c95a2b", "#eda100", "#a86f00"],
  controlnet: ["#008300", "#0f766e", "#199e70", "#0b5d56"],
  vae: ["#1baf7a"],
  other: ["#8a8a8a"],
};
const FALLBACK_COMPONENT_COLORS = ["#4a3aa7", "#6b4fc1", "#315f8c", "#736b2f"];
const PEAK_OVERHEAD_COLOR = "#777";

const ROLE_LABELS = {
  denoiser: "Denoiser",
  vae: "VAE",
  text_encoder: "Text encoder",
  controlnet: "ControlNet",
  other: "Other",
};

export default function MemoryBreakdown(container, props) {
  const wrapper = document.createElement("div");
  wrapper.className = "memory-breakdown-widget nodrag nowheel";
  wrapper.style.cssText = `
    color: inherit;
    background: transparent;
    border: 1px solid currentColor;
    border-radius: 7px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 10px;
    width: 100%;
    font: 12px system-ui, -apple-system, "Segoe UI", sans-serif;
  `;
  const darkMode = document.createElement("style");
  darkMode.textContent = `
    .memory-breakdown-widget {
      --surface: transparent;
      --surface-raised: rgba(0, 0, 0, 0.06);
      --ink: currentColor;
      --secondary-ink: currentColor;
      --muted-ink: currentColor;
      --line: currentColor;
      color-scheme: normal;
    }

    @media (prefers-color-scheme: dark) {
      :root:not([data-theme="light"]) .memory-breakdown-widget {
        --surface: #1a1a19;
        --surface-raised: #242422;
        --ink: #ffffff;
        --secondary-ink: #c3c2b7;
        --muted-ink: #898781;
        --line: #383835;
        color-scheme: dark;
      }
    }

    :root[data-theme="dark"] .memory-breakdown-widget,
    html.dark .memory-breakdown-widget,
    body.dark .memory-breakdown-widget,
    [data-theme="dark"] .memory-breakdown-widget,
    [data-color-scheme="dark"] .memory-breakdown-widget,
    [class*="dark"] .memory-breakdown-widget {
      --surface: #1a1a19;
      --surface-raised: #242422;
      --ink: #ffffff;
      --secondary-ink: #c3c2b7;
      --muted-ink: #898781;
      --line: #383835;
      color-scheme: dark;
    }

    .memory-info-tooltip {
      display: none;
      position: absolute;
      left: 50%;
      top: calc(100% + 6px);
      transform: translateX(-50%);
      z-index: 10;
      min-width: 220px;
      max-width: 320px;
      padding: 7px 9px;
      border: 1px solid var(--line);
      border-radius: 5px;
      background: var(--surface-raised);
      color: var(--ink);
      box-shadow: 0 3px 12px rgba(0, 0, 0, 0.28);
      font-size: 11px;
      line-height: 1.4;
      text-align: left;
      white-space: pre-line;
      pointer-events: none;
    }

    .memory-info:hover .memory-info-tooltip,
    .memory-info:focus-within .memory-info-tooltip {
      display: block;
    }

    :root[data-theme="light"] .memory-breakdown-widget,
    html.light .memory-breakdown-widget,
    body.light .memory-breakdown-widget,
    [data-theme="light"] .memory-breakdown-widget,
    [data-color-scheme="light"] .memory-breakdown-widget,
    [class*="light"] .memory-breakdown-widget {
      --surface: transparent;
      --surface-raised: rgba(0, 0, 0, 0.06);
      --ink: currentColor;
      --secondary-ink: currentColor;
      --muted-ink: currentColor;
      --line: currentColor;
      color-scheme: light;
    }
  `;
  container.replaceChildren(wrapper);

  function numberOrZero(value) {
    const number = Number(value);
    if (!Number.isFinite(number) || number < 0) return 0;
    return number;
  }

  function formatBytes(value) {
    const bytes = numberOrZero(value);
    if (bytes < 1024) return `${Math.round(bytes)} B`;
    const units = ["KiB", "MiB", "GiB", "TiB"];
    let scaled = bytes;
    let unitIndex = -1;
    while (scaled >= 1024 && unitIndex < units.length - 1) {
      scaled /= 1024;
      unitIndex += 1;
    }
    const precision = scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2;
    return `${scaled.toFixed(precision)} ${units[unitIndex]}`;
  }

  function text(value, fallback = "") {
    return typeof value === "string" ? value : fallback;
  }

  function componentColor(component) {
    const role = component.role;
    const id = component.id;
    const family = COMPONENT_COLOR_FAMILIES[role];
    if (!family) {
      const hash = [...id].reduce((value, character) => value * 31 + character.charCodeAt(0), 0);
      return FALLBACK_COMPONENT_COLORS[Math.abs(hash) % FALLBACK_COMPONENT_COLORS.length];
    }

    if (family.length === 1) return family[0];
    if (role === "other") return family[0];

    let shadeIndex = 0;
    if (role === "denoiser") {
      if (id === "unet") {
        shadeIndex = 1;
      } else {
        const match = id.match(/^transformer_(\d+)$/);
        if (match) shadeIndex = Math.max(0, Number(match[1]) - 1);
      }
    } else if (role === "text_encoder") {
      const match = id.match(/^text_encoder(?:_(\d+))?$/);
      if (match?.[1]) shadeIndex = Number(match[1]) - 1;
    } else if (role === "controlnet") {
      const match = id.match(/^controlnet(?:_(\d+))?$/);
      if (match?.[1]) shadeIndex = Number(match[1]);
    }

    return family[shadeIndex % family.length];
  }

  function createTextElement(tag, value, style = "") {
    const element = document.createElement(tag);
    element.textContent = value;
    if (style) element.style.cssText = style;
    return element;
  }

  function normalizeReport(raw) {
    if (!raw || typeof raw !== "object") {
      return { status: "waiting", components: [], total: 0, peak: 0, warnings: [] };
    }
    const components = Array.isArray(raw.components)
      ? raw.components.map((component, index) => {
          const total = numberOrZero(component?.total_bytes);
          return {
            id: text(component?.id, `component-${index}`),
            label: text(component?.label, text(component?.id, `Component ${index + 1}`)),
            role: text(component?.role, "other"),
            total,
            weight: numberOrZero(component?.weight_bytes),
            activation: numberOrZero(component?.activation_bytes),
            estimated: component?.is_estimated === true,
            warning: text(component?.warning),
            tooltip: text(component?.tooltip),
          };
        })
      : [];
    const componentTotal = components.reduce((sum, component) => sum + component.total, 0);
    const declaredTotal = numberOrZero(raw.total_bytes);
    const total = Math.max(componentTotal, declaredTotal);
    const warnings = Array.isArray(raw.warnings) ? raw.warnings.filter((warning) => typeof warning === "string") : [];
    if (declaredTotal > 0 && declaredTotal < componentTotal) {
      warnings.push("Component totals exceed the declared total; the bar is scaled to the component sum.");
    }
    if (declaredTotal > componentTotal) {
      components.push({
        id: "unattributed",
        label: "Unattributed",
        role: "other",
        total: declaredTotal - componentTotal,
        weight: 0,
        activation: 0,
        estimated: true,
        warning: "Declared total not assigned to a named component.",
      });
    }
    const totalDeviceMemory = numberOrZero(raw.total_device_memory_bytes);
    const freeDeviceMemory = numberOrZero(raw.free_device_memory_bytes);
    const device = text(raw.device, "unknown");
    return {
      status: text(raw.status, "waiting"),
      pipelineName: text(raw.pipeline_name, "Memory estimate"),
      components,
      loraAdapters: Array.isArray(raw.lora_adapters) ? raw.lora_adapters : [],
      total,
      peak: numberOrZero(raw.estimated_peak_bytes),
      device,
      totalDeviceMemory,
      freeDeviceMemory,
      headroomPercent: numberOrZero(raw.headroom_percent),
      optimizationSummary: text(raw.optimization_summary, "no additional memory optimizations"),
      warnings,
      message: text(raw.message),
    };
  }

  function addHeader(report) {
    const header = document.createElement("div");
    header.style.cssText = "display:flex; justify-content:space-between; gap:8px; align-items:baseline;";
    header.appendChild(createTextElement("strong", report.pipelineName, "font-size:13px; font-weight:650;"));
    wrapper.appendChild(header);
  }

  function addSummary(report) {
    const summary = document.createElement("div");
    summary.style.cssText = "display:flex; flex-wrap:wrap; gap:12px; color:var(--secondary-ink);";
    const componentTotal = report.components.reduce((sum, component) => sum + component.total, 0);
    const componentStat = document.createElement("span");
    componentStat.appendChild(createTextElement("b", formatBytes(componentTotal), "color:var(--ink); font-size:15px;"));
    componentStat.appendChild(createTextElement("span", " component total", "margin-left:4px;"));
    summary.appendChild(componentStat);
    const peakStat = document.createElement("span");
    peakStat.style.cssText = "display:inline-flex; align-items:center; gap:4px;";
    peakStat.appendChild(createTextElement("b", formatBytes(report.peak), "color:var(--ink); font-size:15px;"));
    peakStat.appendChild(createTextElement("span", " estimated peak"));

    const info = document.createElement("span");
    info.className = "memory-info";
    info.setAttribute("aria-label", "Optimization settings");
    info.tabIndex = 0;
    info.style.cssText = "position:relative; display:inline-flex; align-items:center; font-size:14px; line-height:1; color:var(--secondary-ink);";
    info.appendChild(createTextElement("span", "ⓘ", "display:inline-flex; align-items:center;"));
    const tooltip = createTextElement(
      "span",
      `Includes ${report.headroomPercent}% safety headroom.\n${report.optimizationSummary}`,
      "",
    );
    tooltip.className = "memory-info-tooltip";
    info.appendChild(tooltip);
    peakStat.appendChild(info);
    summary.appendChild(peakStat);
    wrapper.appendChild(summary);
  }

  function addBar(report) {
    const bar = document.createElement("div");
    bar.setAttribute("role", "img");
    bar.setAttribute("aria-label", "Memory usage breakdown by component");
    bar.style.cssText = "display:flex; gap:2px; min-height:24px; background:var(--surface-raised); border-radius:4px; overflow:hidden;";
    const hasDeviceCapacity = report.totalDeviceMemory > 0;
    const barTotal = hasDeviceCapacity ? report.totalDeviceMemory : report.total;
    report.components.forEach((component) => {
      const segment = document.createElement("div");
      const share = barTotal > 0 ? (component.total / barTotal) * 100 : 0;
      segment.tabIndex = 0;
      const shortTitle = `${component.label}: ${formatBytes(component.total)} (${share.toFixed(1)}%)`;
      segment.title = component.tooltip ? `${component.tooltip}\n${shortTitle}` : shortTitle;
      segment.setAttribute("aria-label", segment.title);
      segment.style.cssText = `flex: 0 0 ${Math.max(0, share)}%; min-width: ${share > 0 ? "2px" : "0"}; background:${componentColor(component)}; border-radius:3px; outline-offset:2px;`;
      bar.appendChild(segment);
    });
    if (report.components.length === 0) {
      bar.appendChild(createTextElement("span", "No component data", "align-self:center; padding:4px 8px; color:var(--muted-ink);"));
    }
    const componentTotal = report.components.reduce((sum, component) => sum + component.total, 0);
    const peakOverhead = Math.max(0, Math.min(report.peak, report.totalDeviceMemory) - componentTotal);
    if (hasDeviceCapacity && peakOverhead > 0) {
      const overheadShare = (peakOverhead / report.totalDeviceMemory) * 100;
      const overheadSegment = document.createElement("div");
      overheadSegment.tabIndex = 0;
      overheadSegment.style.cssText = `flex: 0 0 ${overheadShare}%; min-width:2px; background:${PEAK_OVERHEAD_COLOR}; border-radius:3px; outline-offset:2px;`;
      overheadSegment.title = `Peak overhead: ${overheadShare.toFixed(1)}% (${formatBytes(peakOverhead)})`;
      overheadSegment.setAttribute("aria-label", overheadSegment.title);
      bar.appendChild(overheadSegment);
    }
    const remainingDeviceMemory = report.totalDeviceMemory - report.peak;
    const capacityLabel = report.device === "mps" ? "MPS memory budget" : "VRAM";
    if (hasDeviceCapacity && remainingDeviceMemory > 0) {
      const freeSegment = document.createElement("div");
      const freeShare = (remainingDeviceMemory / report.totalDeviceMemory) * 100;
      freeSegment.style.cssText = `flex: 0 0 ${freeShare}%; min-width:2px; background:#555; border-radius:3px;`;
      freeSegment.title = `Estimated free ${capacityLabel} after peak: ${freeShare.toFixed(1)}% (${formatBytes(remainingDeviceMemory)})`;
      freeSegment.setAttribute("aria-label", freeSegment.title);
      bar.appendChild(freeSegment);
    }
    wrapper.appendChild(bar);

    if (hasDeviceCapacity) {
      const deviceInfo = document.createElement("div");
      deviceInfo.style.cssText = "display:flex; justify-content:space-between; gap:8px; color:var(--secondary-ink); font-size:11px;";
      const totalLabel = report.device === "mps" ? "Total MPS memory budget" : "Total VRAM";
      deviceInfo.appendChild(createTextElement("span", `${totalLabel} ${formatBytes(report.totalDeviceMemory)}`));
      if (remainingDeviceMemory > 0) {
        const freeShare = (remainingDeviceMemory / report.totalDeviceMemory) * 100;
        deviceInfo.appendChild(createTextElement("span", `Estimated free ${freeShare.toFixed(1)}% · ${formatBytes(remainingDeviceMemory)}`));
      } else {
        deviceInfo.appendChild(createTextElement("span", "Estimated free 0%"));
      }
      wrapper.appendChild(deviceInfo);
    }
  }

  function addDetails(report) {
    const details = document.createElement("div");
    details.style.cssText = "display:flex; flex-direction:column; gap:5px;";
    report.components.forEach((component) => {
      const row = document.createElement("div");
      row.tabIndex = 0;
      const shortTitle = `${component.label}: ${formatBytes(component.total)}`;
      row.title = component.tooltip ? `${component.tooltip}\n${shortTitle}` : shortTitle;
      row.setAttribute("aria-label", row.title);
      row.style.cssText = "display:grid; grid-template-columns:minmax(0, 1fr) auto; gap:4px 8px; border-top:1px solid var(--line); padding-top:5px; outline-offset:2px;";
      const label = document.createElement("div");
      label.style.cssText = "min-width:0; display:flex; align-items:center; gap:6px;";
      const swatch = document.createElement("span");
      swatch.setAttribute("aria-hidden", "true");
      swatch.style.cssText = `width:9px; height:9px; flex:0 0 9px; border-radius:2px; background:${componentColor(component)};`;
      label.appendChild(swatch);
      label.appendChild(createTextElement("span", component.label, "overflow:hidden; text-overflow:ellipsis; white-space:nowrap;"));
      label.appendChild(createTextElement("small", ROLE_LABELS[component.role] || component.role, "color:var(--muted-ink);"));
      row.appendChild(label);
      row.appendChild(createTextElement("strong", formatBytes(component.total), "font-variant-numeric:tabular-nums; white-space:nowrap;"));
      const breakdown = createTextElement("small", `Weights ${formatBytes(component.weight)} · Activations ${formatBytes(component.activation)}`, "grid-column:1 / -1; color:var(--secondary-ink); padding-left:15px;");
      row.appendChild(breakdown);
      if (component.estimated || component.warning) {
        row.appendChild(createTextElement("small", `[Estimated] ${component.warning || "This component uses an approximation."}`, "grid-column:1 / -1; color:var(--secondary-ink); padding-left:15px;"));
      }
      details.appendChild(row);
    });
    wrapper.appendChild(details);
  }

  function addNotes(report) {
    const notes = [...report.warnings];
    if (report.message) notes.push(report.message);
    if (notes.length === 0) return;
    const noteBox = document.createElement("div");
    noteBox.style.cssText = "border-top:1px solid var(--line); padding-top:7px; color:var(--secondary-ink); font-size:11px; display:flex; flex-direction:column; gap:3px;";
    notes.forEach((note) => noteBox.appendChild(createTextElement("div", note)));
    wrapper.appendChild(noteBox);
  }

  function render(rawValue) {
    wrapper.replaceChildren(darkMode);
    const report = normalizeReport(rawValue);
    addHeader(report);
    if (report.status === "ready") {
      addSummary(report);
      addBar(report);
      addDetails(report);
    } else {
      wrapper.appendChild(
        createTextElement(
          "div",
          report.message || "Run the node to calculate a memory estimate.",
          "color:var(--secondary-ink); padding:8px 0;",
        ),
      );
    }
    addNotes(report);
  }

  render(props?.value);

  return {
    update(nextProps) {
      render(nextProps?.value);
    },
    cleanup() {
      container.replaceChildren();
    },
  };
}
