/* SWI-TTA interactive project page logic (zero external dependencies). */

const FALLBACK_COHORTS = [
  { group: "validation", display_label: "LB-Matched (Val, OpenSWI)", regime: "shallow", is_field: false, all_checkpoints: { step50_delta_rmse_pct: -59.14, holdout_es_delta_rmse_pct: -58.00, protocol_delta_rmse_pct: -57.46, routed_pct: 93.58, reverted_pct_of_all: 0.00, disba_call_savings_pct: -11.17 } },
  { group: "validation", display_label: "Shallow Bench. (Val, OpenSWI)", regime: "shallow", is_field: false, all_checkpoints: { step50_delta_rmse_pct: -3.04, holdout_es_delta_rmse_pct: -10.99, protocol_delta_rmse_pct: -23.16, routed_pct: 36.73, reverted_pct_of_all: 0.11, disba_call_savings_pct: 50.90 } },
  { group: "validation", display_label: "Deep Bench. (Val, OpenSWI)", regime: "deep", is_field: false, all_checkpoints: { step50_delta_rmse_pct: 8.01, holdout_es_delta_rmse_pct: -3.85, protocol_delta_rmse_pct: -3.51, routed_pct: 12.67, reverted_pct_of_all: 0.15, disba_call_savings_pct: 77.17 } },
  { group: "primary_synthetic", display_label: "LB-Matched (Test, OpenSWI)", regime: "shallow", is_field: false, all_checkpoints: { step50_delta_rmse_pct: -52.01, holdout_es_delta_rmse_pct: -50.51, protocol_delta_rmse_pct: -50.18, routed_pct: 94.45, reverted_pct_of_all: 0.00, disba_call_savings_pct: -12.13 } },
  { group: "primary_synthetic", display_label: "Shallow Bench. (Test, OpenSWI)", regime: "shallow", is_field: false, all_checkpoints: { step50_delta_rmse_pct: -6.74, holdout_es_delta_rmse_pct: null, protocol_delta_rmse_pct: -25.81, routed_pct: 37.57, reverted_pct_of_all: 0.12, disba_call_savings_pct: 50.01 } },
  { group: "primary_synthetic", display_label: "Deep Bench. (Test, OpenSWI)", regime: "deep", is_field: false, all_checkpoints: { step50_delta_rmse_pct: 21.86, holdout_es_delta_rmse_pct: -3.29, protocol_delta_rmse_pct: -3.30, routed_pct: 12.97, reverted_pct_of_all: 0.13, disba_call_savings_pct: 76.86 } },
  { group: "diagnostic", display_label: "Sh. Multi-Struct. (OpenSWI)", regime: "shallow", is_field: false, all_checkpoints: { step50_delta_rmse_pct: -4.48, holdout_es_delta_rmse_pct: -12.15, protocol_delta_rmse_pct: -22.86, routed_pct: 42.95, reverted_pct_of_all: 0.17, disba_call_savings_pct: 44.19 } },
  { group: "diagnostic", display_label: "Sh. Prior-Shift (OpenSWI)", regime: "shallow", is_field: false, all_checkpoints: { step50_delta_rmse_pct: -39.07, holdout_es_delta_rmse_pct: -39.34, protocol_delta_rmse_pct: -40.48, routed_pct: 72.96, reverted_pct_of_all: 0.04, disba_call_savings_pct: 11.35 } },
  { group: "diagnostic", display_label: "Deep Lith. (OpenSWI)", regime: "deep", is_field: false, all_checkpoints: { step50_delta_rmse_pct: 10.59, holdout_es_delta_rmse_pct: -1.50, protocol_delta_rmse_pct: -3.63, routed_pct: 24.22, reverted_pct_of_all: 0.05, disba_call_savings_pct: 64.56 } },
  { group: "diagnostic", display_label: "Sh. Multi-Struct. (DispFormer)", regime: "shallow", is_field: false, all_checkpoints: { step50_delta_rmse_pct: -13.87, holdout_es_delta_rmse_pct: -15.66, protocol_delta_rmse_pct: -15.49, routed_pct: 63.05, reverted_pct_of_all: 0.05, disba_call_savings_pct: 22.17 } },
  { group: "diagnostic", display_label: "Sh. Prior-Shift (DispFormer)", regime: "shallow", is_field: false, all_checkpoints: { step50_delta_rmse_pct: -19.81, holdout_es_delta_rmse_pct: -18.81, protocol_delta_rmse_pct: -17.48, routed_pct: 71.71, reverted_pct_of_all: 0.01, disba_call_savings_pct: 12.68 } },
  { group: "diagnostic", display_label: "Deep Lith. (DispFormer)", regime: "deep", is_field: false, all_checkpoints: { step50_delta_rmse_pct: -1.00, holdout_es_delta_rmse_pct: -7.42, protocol_delta_rmse_pct: -5.45, routed_pct: 33.69, reverted_pct_of_all: 0.07, disba_call_savings_pct: 54.22 } },
  { group: "field_report", display_label: "LongBeach Report (OpenSWI)", regime: "shallow", is_field: true, all_checkpoints: { ho_median_zs: 0.0375, ho_median_protocol: 0.0082, r_ref_zs: 0.551, r_ref_protocol: 0.713, routed_pct: 98.14, reverted_pct_of_all: 0.04, disba_call_savings_pct: -16.15 } },
  { group: "field_report", display_label: "CSRM Report (OpenSWI)", regime: "deep", is_field: true, all_checkpoints: { ho_median_zs: 0.0123, ho_median_protocol: 0.0110, r_ref_zs: 0.686, r_ref_protocol: 0.707, routed_pct: 21.64, reverted_pct_of_all: 0.02, disba_call_savings_pct: 67.46 } }
];

const CSRM_DEBIAS = {
  "1.0x": {
    sigma: "1.356%",
    zs: 70.9,
    hes: 21.6,
    flag: 7.4,
    note: "Calibrated 7-knot classical noise floor (default): protects 70.9% of grid points as zero-shot and cuts exact disba calls by +67.5%."
  },
  "1.5x": {
    sigma: "0.904%",
    zs: 44.0,
    hes: 42.7,
    flag: 13.3,
    note: "Downward 1.5x debiasing sensitivity check: 44.0% remain zero-shot while 42.7% undergo Holdout ES refinement and 13.3% are flagged >3.0."
  },
  "1.87x": {
    sigma: "0.725%",
    zs: 26.0,
    hes: 54.7,
    flag: 19.3,
    note: "Maximal 1.87x debiasing (matching synthetic 1.82-1.92x upward bias): 26.0% remain zero-shot, 54.7% refine with Holdout ES, and 19.3% are flagged."
  }
};

function initTabs() {
  document.querySelectorAll("[data-tab-group]").forEach((groupEl) => {
    const buttons = groupEl.querySelectorAll(".tab-btn");
    const panes = groupEl.querySelectorAll(".tab-pane");
    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = btn.getAttribute("data-tab-target");
        buttons.forEach((b) => b.classList.toggle("active", b === btn));
        panes.forEach((pane) => {
          pane.classList.toggle("active", pane.getAttribute("data-tab-id") === target);
        });
      });
    });
  });
}

function initLightbox() {
  const modal = document.getElementById("lightbox");
  const modalImg = document.getElementById("lightbox-img");
  if (!modal || !modalImg) return;

  document.querySelectorAll(".figure-img").forEach((img) => {
    img.addEventListener("click", () => {
      modalImg.src = img.currentSrc || img.src;
      modalImg.alt = img.alt;
      modal.classList.add("open");
    });
  });

  modal.addEventListener("click", () => modal.classList.remove("open"));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") modal.classList.remove("open");
  });
}

function initCopyButtons() {
  document.querySelectorAll("[data-copy-target]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const targetId = btn.getAttribute("data-copy-target");
      const el = document.getElementById(targetId);
      if (!el) return;
      try {
        await navigator.clipboard.writeText(el.innerText.trim());
        const old = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(() => { btn.textContent = old; }, 1600);
      } catch {
        btn.textContent = "Copy failed";
      }
    });
  });
}

function initCsrmDebiasWidget() {
  const buttons = document.querySelectorAll("[data-debias]");
  const segZs = document.getElementById("seg-zs");
  const segHes = document.getElementById("seg-hes");
  const segFlag = document.getElementById("seg-flag");
  const noteEl = document.getElementById("debias-note");
  const sigmaEl = document.getElementById("debias-sigma");
  if (!segZs) return;

  function applyScale(key) {
    const d = CSRM_DEBIAS[key];
    if (!d) return;
    segZs.style.width = `${d.zs}%`;
    segZs.textContent = `Zero-Shot ${d.zs.toFixed(1)}%`;
    segHes.style.width = `${d.hes}%`;
    segHes.textContent = `Holdout ES ${d.hes.toFixed(1)}%`;
    segFlag.style.width = `${d.flag}%`;
    segFlag.textContent = d.flag >= 10 ? `Flag ${d.flag.toFixed(1)}%` : `${d.flag.toFixed(1)}%`;
    sigmaEl.textContent = d.sigma;
    noteEl.textContent = d.note;
    buttons.forEach((b) => b.classList.toggle("active", b.getAttribute("data-debias") === key));
  }

  buttons.forEach((b) => b.addEventListener("click", () => applyScale(b.getAttribute("data-debias"))));
  applyScale("1.0x");
}

function fmtSigned(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "--";
  return `${v >= 0 ? "+" : ""}${Number(v).toFixed(2)}%`;
}

function groupBadge(grp) {
  const map = {
    validation: "Val",
    primary_synthetic: "Test",
    diagnostic: "Diag",
    field_report: "Field"
  };
  return map[grp] || grp;
}

async function initProtocolTable() {
  const tbody = document.getElementById("protocol-tbody");
  const filterBtns = document.querySelectorAll("[data-cohort-filter]");
  if (!tbody) return;

  let cohorts = FALLBACK_COHORTS;
  for (const url of ["assets/data/operational_protocol_audit.json", "../results_summary/operational_protocol_audit.json"]) {
    try {
      const resp = await fetch(url);
      if (resp.ok) {
        const doc = await resp.json();
        if (Array.isArray(doc.cohorts) && doc.cohorts.length === 14) {
          cohorts = doc.cohorts;
          break;
        }
      }
    } catch {
      // Fallback to next candidate or embedded snapshot
    }
  }

  function render(filterKey) {
    tbody.innerHTML = "";
    const subset = filterKey === "all" ? cohorts : cohorts.filter((c) => c.group === filterKey);
    subset.forEach((c) => {
      const ac = c.all_checkpoints;
      const tr = document.createElement("tr");
      const rule = c.regime === "shallow" ? "r0 >= 2.0 + HES" : "[1.25, 3.0] + HES";
      const routed = `${ac.routed_pct.toFixed(1)}% (${ac.reverted_pct_of_all.toFixed(2)}%)`;
      const savedCls = ac.disba_call_savings_pct >= 0 ? "val-good" : "";
      const savedTxt = `${ac.disba_call_savings_pct >= 0 ? "+" : ""}${ac.disba_call_savings_pct.toFixed(1)}%`;

      if (c.is_field) {
        const ho0 = (ac.ho_median_zs * 100).toFixed(2);
        const ho1 = (ac.ho_median_protocol * 100).toFixed(2);
        tr.innerHTML = `
          <td><span class="badge-group">${groupBadge(c.group)}</span>${c.display_label}</td>
          <td><code>${rule}</code></td>
          <td colspan="3" class="val-good">HO ${ho0}% &rarr; ${ho1}% (r_ref ${ac.r_ref_zs.toFixed(3)} &rarr; ${ac.r_ref_protocol.toFixed(3)})</td>
          <td>${routed}</td>
          <td class="${savedCls}">${savedTxt}</td>
        `;
      } else {
        const s50Cls = ac.step50_delta_rmse_pct > 0 ? "val-bad" : "";
        const protoCls = ac.protocol_delta_rmse_pct < 0 ? "val-good" : "val-bad";
        tr.innerHTML = `
          <td><span class="badge-group">${groupBadge(c.group)}</span>${c.display_label}</td>
          <td><code>${rule}</code></td>
          <td class="${s50Cls}">${fmtSigned(ac.step50_delta_rmse_pct)}</td>
          <td>${fmtSigned(ac.holdout_es_delta_rmse_pct)}</td>
          <td class="${protoCls}">${fmtSigned(ac.protocol_delta_rmse_pct)}</td>
          <td>${routed}</td>
          <td class="${savedCls}">${savedTxt}</td>
        `;
      }
      tbody.appendChild(tr);
    });
  }

  filterBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      filterBtns.forEach((b) => b.classList.toggle("active", b === btn));
      render(btn.getAttribute("data-cohort-filter"));
    });
  });

  render("all");
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initLightbox();
  initCopyButtons();
  initCsrmDebiasWidget();
  initProtocolTable();
});
