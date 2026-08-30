/* CryoZeta local server -- lightweight vanilla JS, no build step. */

function initNewJobForm() {
  const config = window.CRYOZETA_CONFIG || {};
  const container = document.getElementById('sequence-rows');
  const template = document.getElementById('sequence-template');
  const addButton = document.getElementById('add-sequence');
  const summary = document.getElementById('length-summary');
  const runMode = document.getElementById('run_mode');
  const modeHint = document.getElementById('mode-hint');
  const inferenceMode = document.getElementById('inference_mode');
  const overwriteWrap = document.getElementById('overwrite-wrap');
  if (!container || !template) return;

  // The user has not touched the pipeline selector yet, so we may auto-switch.
  let modeIsAuto = true;
  runMode.addEventListener('change', () => { modeIsAuto = false; refresh(); });

  const ALPHABETS = {
    proteinChain: /[^ACDEFGHIKLMNPQRSTVWYX]/g,
    dnaSequence: /[^ACGTN]/g,
    rnaSequence: /[^ACGUN]/g
  };

  function cleanSequence(text, type) {
    return text
      .replace(/^\s*>.*$/gm, '')      // FASTA headers
      .replace(/[\s\d*\-.]/g, '')     // whitespace, numbering, gaps, stops
      .toUpperCase()
      .replace(ALPHABETS[type] || /$^/g, '');
  }

  function addRow() {
    const node = template.content.cloneNode(true);
    container.appendChild(node);
    const row = container.lastElementChild;

    row.querySelector('.remove-row').addEventListener('click', () => {
      row.remove();
      refresh();
    });
    row.querySelector('.seq-type').addEventListener('change', refresh);
    row.querySelector('.seq-count').addEventListener('input', refresh);
    row.querySelector('.seq-value').addEventListener('input', refresh);
    refresh();
  }

  function refresh() {
    const rows = Array.from(container.querySelectorAll('.seq-row'));
    let total = 0;
    const proteinSequences = new Set();

    rows.forEach((row, index) => {
      row.querySelector('.row-number').textContent = index + 1;

      const type = row.querySelector('.seq-type').value;
      const raw = row.querySelector('.seq-value').value;
      const cleaned = cleanSequence(raw, type);
      const count = Math.max(1, parseInt(row.querySelector('.seq-count').value, 10) || 1);

      total += cleaned.length * count;
      if (type === 'proteinChain' && cleaned.length) proteinSequences.add(cleaned);

      row.querySelector('.seq-length').textContent =
        `${cleaned.length} residues${count > 1 ? ` x ${count} copies` : ''}`;

      // DNA needs no MSA at all; hide the whole block for it.
      const msaBlock = row.querySelector('.msa-block');
      msaBlock.hidden = (type === 'dnaSequence');
      row.querySelector('.msa-required').textContent =
        type === 'rnaSequence' ? `Must contain ${config.msaRna}` : '';
    });

    // Mirrors CryoZeta: a pairing MSA is required only when there are two or
    // more *distinct* protein sequences.
    const needsPairing = proteinSequences.size >= 2;
    rows.forEach((row) => {
      if (row.querySelector('.seq-type').value !== 'proteinChain') return;
      row.querySelector('.msa-required').textContent = needsPairing
        ? `Must contain ${config.msaProtein} and ${config.msaProteinPairing}`
        : `Must contain ${config.msaProtein}`;
    });

    const isLarge = total > config.threshold;
    if (modeIsAuto) runMode.value = isLarge ? 'large' : 'standard';

    const usingLarge = runMode.value === 'large';
    inferenceMode.disabled = usingLarge;
    if (overwriteWrap) overwriteWrap.style.opacity = usingLarge ? '.5' : '1';

    modeHint.textContent = isLarge
      ? `${total} residues is above the ${config.threshold} threshold -- large/cycle mode recommended.`
      : `${total} residues is within the ${config.threshold} standard-mode threshold.`;

    if (summary) {
      summary.textContent = rows.length
        ? `${rows.length} chain entr${rows.length === 1 ? 'y' : 'ies'}, ` +
          `${total} residues/nucleotides total` +
          (needsPairing ? ' -- pairing MSA required (2+ distinct protein sequences).' : '.')
        : 'No sequences added yet.';
    }
  }

  addButton.addEventListener('click', addRow);
  addRow();
}

/* ---------- job detail: poll status and tail the log ---------- */
function initJobPolling(jobId, terminal) {
  const logBox = document.getElementById('logbox');
  const statusBadge = document.getElementById('status-badge');
  let stopped = terminal;

  async function tick() {
    if (stopped) return;
    try {
      const [statusResponse, logResponse] = await Promise.all([
        fetch(`/jobs/${jobId}/status`),
        fetch(`/jobs/${jobId}/log`)
      ]);
      if (statusResponse.ok) {
        const data = await statusResponse.json();
        if (statusBadge) {
          statusBadge.textContent = data.status;
          statusBadge.className = 'badge badge-' + badgeClass(data.status);
        }
        updateStages(data.stage);
        if (data.terminal) {
          stopped = true;
          // Reload once so the page picks up results links and final timings.
          setTimeout(() => window.location.reload(), 800);
        }
      }
      if (logResponse.ok && logBox) {
        const pinned =
          logBox.scrollTop + logBox.clientHeight >= logBox.scrollHeight - 40;
        logBox.textContent = await logResponse.text();
        if (pinned) logBox.scrollTop = logBox.scrollHeight;
      }
    } catch (err) {
      /* transient fetch failure -- try again on the next tick */
    }
    if (!stopped) setTimeout(tick, 3000);
  }

  function badgeClass(status) {
    return {
      queued: 'queued', running: 'running', completed: 'ok',
      failed: 'error', cancelled: 'muted', interrupted: 'warn'
    }[status] || 'muted';
  }

  function updateStages(current) {
    const items = Array.from(document.querySelectorAll('.stages li'));
    const currentIndex = items.findIndex((li) => li.dataset.stage === current);
    items.forEach((li, index) => {
      li.classList.remove('done', 'active', 'pending');
      if (current === 'done' || (currentIndex >= 0 && index < currentIndex)) {
        li.classList.add('done');
        li.querySelector('.dot').textContent = '✓';
      } else if (index === currentIndex) {
        li.classList.add('active');
      } else {
        li.classList.add('pending');
      }
    });
  }

  if (logBox) logBox.scrollTop = logBox.scrollHeight;
  tick();
}

/* ---------- results: optional 3Dmol.js preview ---------- */
function initViewer(url) {
  const target = document.getElementById('viewer');
  const status = document.getElementById('viewer-status');
  if (!target) return;

  if (typeof $3Dmol === 'undefined') {
    // The viewer is a convenience layer loaded from a CDN. On an air-gapped
    // workstation it simply will not be there, which must not break the page.
    status.textContent =
      '3D preview unavailable (viewer library could not be loaded). ' +
      'Download the mmCIF and open it in ChimeraX or PyMOL.';
    target.style.display = 'none';
    return;
  }

  const viewer = $3Dmol.createViewer(target, { backgroundColor: '#0b0d10' });
  fetch(url)
    .then((response) => response.text())
    .then((data) => {
      viewer.addModel(data, 'cif');
      viewer.setStyle({}, { cartoon: { colorscheme: 'chain' } });
      viewer.zoomTo();
      viewer.render();
      status.textContent = '';
    })
    .catch(() => {
      status.textContent = 'Could not load the model for preview.';
    });
}
