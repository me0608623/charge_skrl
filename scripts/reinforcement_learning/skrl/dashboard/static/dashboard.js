/* Charge-SKRL Training Dashboard — Frontend Logic */

// ── Tab switching ──
document.addEventListener('DOMContentLoaded', () => {
  const btns = document.querySelectorAll('.tab-btn');
  const panels = document.querySelectorAll('.tab-panel');

  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      btns.forEach(b => b.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(target).classList.add('active');

      // Lazy-init Plotly charts on first tab show
      if (window._chartInits && window._chartInits[target]) {
        window._chartInits[target]();
        delete window._chartInits[target];
      }
    });
  });

  // Copy button for code blocks
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const code = btn.parentElement.querySelector('code') || btn.parentElement;
      const text = code.textContent.replace(btn.textContent, '').trim();
      navigator.clipboard.writeText(text).then(() => {
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = 'Copy', 1500);
      });
    });
  });
});

// ── Plotly dark theme ──
const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: '#292e42',
  plot_bgcolor: '#1a1b26',
  font: { color: '#c0caf5', family: 'Inter, sans-serif', size: 12 },
  margin: { l: 50, r: 20, t: 40, b: 40 },
  xaxis: { gridcolor: '#3b4261', zerolinecolor: '#3b4261' },
  yaxis: { gridcolor: '#3b4261', zerolinecolor: '#3b4261' },
  legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } },
};

const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

function mergeLayout(overrides) {
  return Object.assign({}, JSON.parse(JSON.stringify(PLOTLY_LAYOUT_BASE)), overrides);
}

// ── Chart registry (lazy init) ──
window._chartInits = {};

// ── Curriculum version switcher ──
function switchCurriculumVersion(version) {
  const allTables = document.querySelectorAll('.curriculum-table');
  allTables.forEach(t => t.style.display = 'none');
  const target = document.getElementById('curriculum-table-' + version);
  if (target) target.style.display = '';

  // Re-render charts for this version
  if (window._renderCurriculumCharts) {
    window._renderCurriculumCharts(version);
  }
}
