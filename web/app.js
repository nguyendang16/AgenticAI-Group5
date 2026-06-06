const API = '';

const uploadForm = document.getElementById('upload-form');
const pdfFile = document.getElementById('pdf-file');
const fileName = document.getElementById('file-name');
const submitBtn = document.getElementById('submit-btn');
const submitError = document.getElementById('submit-error');
const progressPanel = document.getElementById('progress-panel');
const statusBadge = document.getElementById('status-badge');
const statusMessage = document.getElementById('status-message');
const annotationCount = document.getElementById('annotation-count');
const toolCalls = document.getElementById('tool-calls');
const toolTimeline = document.getElementById('tool-timeline');
const evaluateBtn = document.getElementById('evaluate-btn');
const evalPanel = document.getElementById('eval-panel');
const evalFramework = document.getElementById('eval-framework');
const evalVerdict = document.getElementById('eval-verdict');
const evalSummary = document.getElementById('eval-summary');
const evalChecklistWrap = document.getElementById('eval-checklist-wrap');
const evalChecklistBody = document.getElementById('eval-checklist-body');
const evalOqiDetails = document.getElementById('eval-oqi-details');
const evalOqi = document.getElementById('eval-oqi');
const evalDetails = document.getElementById('eval-details');
const evalJson = document.getElementById('eval-json');
const evalError = document.getElementById('eval-error');

let pollTimer = null;
let eventsAfter = 0;
let currentJobId = null;

pdfFile.addEventListener('change', () => {
  const file = pdfFile.files?.[0];
  fileName.textContent = file ? file.name : 'Choose PDF…';
});

uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  submitError.hidden = true;

  const file = pdfFile.files?.[0];
  if (!file) {
    submitError.textContent = 'Please select a PDF file.';
    submitError.hidden = false;
    return;
  }

  const form = new FormData();
  form.append('file', file);
  const title = document.getElementById('title-input').value.trim();
  if (title) form.append('title', title);
  const venue = document.getElementById('review-venue')?.value.trim();
  const journal = document.getElementById('review-journal')?.value.trim();
  const domain = document.getElementById('review-domain')?.value.trim();
  const articleType = document.getElementById('review-article-type')?.value.trim();
  if (venue) form.append('review_venue', venue);
  if (journal) form.append('review_journal', journal);
  if (domain) form.append('review_domain', domain);
  if (articleType) form.append('review_article_type', articleType);

  submitBtn.disabled = true;
  submitBtn.textContent = 'Submitting…';

  try {
    const res = await fetch(`${API}/api/jobs`, { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || data.message || 'Submit failed');
    }
    startJob(data.job_id);
  } catch (err) {
    submitError.textContent = err.message || String(err);
    submitError.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Start review';
  }
});

function startJob(jobId) {
  currentJobId = jobId;
  eventsAfter = 0;
  toolTimeline.innerHTML = '';
  progressPanel.hidden = false;
  if (evalPanel) evalPanel.hidden = true;
  if (evaluateBtn) evaluateBtn.hidden = true;
  if (evalError) evalError.hidden = true;
  if (evalChecklistWrap) evalChecklistWrap.hidden = true;
  if (evalOqiDetails) evalOqiDetails.hidden = true;
  statusBadge.textContent = 'queued';
  statusBadge.className = 'status-badge';
  statusMessage.textContent = 'Job created…';

  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => pollJob(jobId), 1500);
  pollJob(jobId);
}

async function pollJob(jobId) {
  try {
    const [statusRes, eventsRes] = await Promise.all([
      fetch(`${API}/api/jobs/${jobId}`),
      fetch(`${API}/api/jobs/${jobId}/events?after=${eventsAfter}`),
    ]);

    if (!statusRes.ok) throw new Error('Failed to load job status');
    const status = await statusRes.json();
    updateStatus(status);

    if (eventsRes.ok) {
      const eventsPayload = await eventsRes.json();
      eventsAfter = eventsPayload.next_after ?? eventsAfter;
      appendEvents(eventsPayload.events || []);
    }

    if (status.status === 'completed') {
      clearInterval(pollTimer);
      pollTimer = null;
      await loadJobEvaluation(jobId);
    } else if (status.status === 'failed') {
      clearInterval(pollTimer);
      pollTimer = null;
      submitError.textContent = status.error || status.message || 'Job failed';
      submitError.hidden = false;
      await loadJobEvaluation(jobId);
    }
  } catch (err) {
    console.error(err);
  }
}

function updateStatus(status) {
  statusBadge.textContent = status.status;
  statusBadge.className = `status-badge ${status.status}`;
  statusMessage.textContent = status.message || '';
  annotationCount.textContent = String(status.annotation_count ?? 0);
  const perTool = status.usage?.tool?.per_tool || {};
  const total = status.usage?.tool?.total_calls ?? Object.values(perTool).reduce((a, b) => a + b, 0);
  toolCalls.textContent = String(total);
}

function appendEvents(events) {
  for (const ev of events) {
    const li = document.createElement('li');
    const label = formatEvent(ev);
    li.innerHTML = label;
    toolTimeline.appendChild(li);
    toolTimeline.scrollTop = toolTimeline.scrollHeight;
  }
}

function formatEvent(ev) {
  const ts = ev.ts ? new Date(ev.ts).toLocaleTimeString() : '';
  const event = ev.event || '';

  if (event === 'tool_call') {
    const detail = formatToolDetail(ev);
    return `<span class="event-meta">${ts}</span> <span class="tool-name">${escapeHtml(ev.tool)}</span>${detail}`;
  }
  if (event === 'annotation_created') {
    return `<span class="event-meta">${ts}</span> <span class="tool-name">annotation</span> page ${ev.page} · ${escapeHtml(ev.object_type || '')}`;
  }
  if (event === 'agent_status_update') {
    return `<span class="event-meta">${ts}</span> <span class="tool-name">status</span> ${escapeHtml(ev.step || '')}`;
  }
  if (event === 'status' || event === 'failed') {
    return `<span class="event-meta">${ts}</span> <strong>${escapeHtml(ev.status || event)}</strong> ${escapeHtml(ev.message || '')}`;
  }
  if (event === 'worker_spawned') {
    return `<span class="event-meta">${ts}</span> Worker started`;
  }
  if (event === 'created') {
    return `<span class="event-meta">${ts}</span> Job created`;
  }
  if (event === 'review_criteria_resolved') {
    const count = ev.criteria_count ?? 0;
    const source = ev.source || ev.skipped || 'none';
    return `<span class="event-meta">${ts}</span> <span class="tool-name">criteria</span> ${count} from ${escapeHtml(source)}`;
  }
  if (event === 'evaluation_completed') {
    const bucket = ev.root_cause_bucket ? ` · ${ev.root_cause_bucket}` : '';
    return `<span class="event-meta">${ts}</span> <span class="tool-name">evaluation</span> completed${escapeHtml(bucket)}`;
  }
  if (event === 'evaluation_failed') {
    return `<span class="event-meta">${ts}</span> <span class="tool-name">evaluation</span> failed ${escapeHtml(ev.error || '')}`;
  }

  return `<span class="event-meta">${ts}</span> ${escapeHtml(event)}`;
}

function formatToolDetail(ev) {
  const parts = [];
  if (ev.query) parts.push(`query: "${truncate(ev.query, 60)}"`);
  if (ev.page != null) parts.push(`p.${ev.page}`);
  if (ev.start_line != null) parts.push(`L${ev.start_line}-${ev.end_line ?? ev.start_line}`);
  if (ev.section_id) parts.push(`section: ${ev.section_id}`);
  if (ev.item_count != null) parts.push(`items: ${ev.item_count}`);
  if (!parts.length) return '';
  return ` <span class="event-meta">(${escapeHtml(parts.join(', '))})</span>`;
}

function truncate(s, n) {
  const t = String(s);
  return t.length > n ? `${t.slice(0, n)}…` : t;
}

function escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function showEvaluateButton(jobId, { alreadyLoaded = false } = {}) {
  if (!evaluateBtn) return;
  evaluateBtn.hidden = false;
  evaluateBtn.textContent = alreadyLoaded ? 'Re-run evaluation' : 'Run evaluation';
  evaluateBtn.onclick = () => runEvaluation(jobId);
}

async function loadJobEvaluation(jobId) {
  if (!evalPanel) return null;
  evalError.hidden = true;

  try {
    const getRes = await fetch(`${API}/api/jobs/${jobId}/evaluation`);
    if (getRes.ok) {
      const data = await getRes.json();
      renderEvaluation(data);
      showEvaluateButton(jobId, { alreadyLoaded: true });
      return data;
    }
    if (getRes.status === 404) {
      return runEvaluation(jobId, { silent: true });
    }
    const payload = await getRes.json().catch(() => ({}));
    throw new Error(payload.detail || payload.message || 'Failed to load evaluation');
  } catch (err) {
    evalError.textContent = err.message || String(err);
    evalError.hidden = false;
    showEvaluateButton(jobId);
    return null;
  }
}

function evalCard(label, value, ok, meta) {
  const cls = ok === true ? 'pass' : ok === false ? 'fail' : '';
  const metaHtml = meta ? `<span class="meta">${escapeHtml(meta)}</span>` : '';
  return `<div class="eval-card ${cls}"><span class="label">${escapeHtml(label)}</span><span class="value">${escapeHtml(value)}</span>${metaHtml}</div>`;
}

function formatMetricValue(value) {
  if (value === true) return 'Yes';
  if (value === false) return 'No';
  if (value == null) return 'n/a';
  if (typeof value === 'number' && value >= 1000) return value.toLocaleString();
  return String(value);
}

function renderFramework(framework) {
  if (!evalFramework || !framework) return;
  const sources = (framework.sources || [])
    .map(
      (src) => `
        <li>
          <strong>${escapeHtml(src.id?.toUpperCase() || '')}</strong>
          ${escapeHtml(src.authors || '')}, ${escapeHtml(src.venue || '')}
          — <em>${escapeHtml(src.citation || '')}</em>
          ${src.pdf_url ? ` · <a href="${escapeHtml(src.pdf_url)}" target="_blank" rel="noopener">PDF</a>` : ''}
        </li>`,
    )
    .join('');
  evalFramework.innerHTML = `
    <p class="eval-framework-name">${escapeHtml(framework.name || 'System evaluation')}</p>
    <ul class="eval-sources">${sources}</ul>
  `;
}

function renderSystemVerdict(systemVerdict) {
  if (!evalVerdict || !systemVerdict) return;
  const pass = systemVerdict.verdict === 'pass';
  const cls = pass ? 'pass' : 'fail';
  const checklist = `${systemVerdict.checklist_passed ?? 0}/${systemVerdict.checklist_total ?? 0} checks`;
  evalVerdict.innerHTML = `
    <div class="eval-verdict-badge ${cls}">
      <span class="eval-verdict-label">System verdict</span>
      <span class="eval-verdict-value">${pass ? 'PASS' : 'FAIL'}</span>
    </div>
    <p class="eval-verdict-note">${escapeHtml(systemVerdict.note || '')} Checklist: ${escapeHtml(checklist)}.</p>
  `;
}

function renderSystemMetrics(systemMetrics) {
  if (!evalSummary || !systemMetrics) return;
  const order = ['M1_sr', 'M2_k_proxy', 'M3_deliverable', 'M4_multi_round', 'M5_wall_clock', 'M6_tokens', 'M7_trace'];
  evalSummary.innerHTML = order
    .map((key) => {
      const metric = systemMetrics[key];
      if (!metric) return '';
      const id = key.replace('_', ' ').toUpperCase();
      return evalCard(
        `${id}: ${metric.label}`,
        formatMetricValue(metric.value),
        metric.pass,
      );
    })
    .join('');
}

function renderChecklist(checklist) {
  if (!evalChecklistWrap || !evalChecklistBody || !checklist?.length) return;
  evalChecklistBody.innerHTML = checklist
    .map((item) => {
      const result = item.pass ? '✓ Pass' : '✗ Fail';
      const cls = item.pass ? 'pass' : 'fail';
      return `<tr class="${cls}">
        <td>${escapeHtml(item.label)}</td>
        <td class="check-result">${result}</td>
        <td>${escapeHtml(item.source)}</td>
        <td class="evidence">${escapeHtml(item.evidence)}</td>
      </tr>`;
    })
    .join('');
  evalChecklistWrap.hidden = false;
}

function renderOqi(oqi) {
  if (!evalOqiDetails || !evalOqi) return;
  if (!oqi) {
    evalOqiDetails.hidden = true;
    return;
  }
  const q4 = oqi.q4_grounding || {};
  evalOqi.innerHTML = [
    evalCard('OQI total', `${oqi.total}/${oqi.max}`, oqi.tier1_quality_pass, 'informational'),
    evalCard('Q1 structure', `${(oqi.q1_structure || {}).score}/2`, null, null),
    evalCard('Q2 coverage', `${(oqi.q2_coverage || {}).score}/2`, null, null),
    evalCard('Q3 specificity', `${(oqi.q3_specificity || {}).score}/2`, null, null),
    evalCard(
      'Q4 grounding sample',
      q4.sample_size ? `${q4.passed}/${q4.sample_size}` : 'n/a',
      q4.sample_size ? q4.passed === q4.sample_size : null,
      null,
    ),
    evalCard('Q5 actionability', `${(oqi.q5_actionability || {}).score}/2`, null, null),
  ].join('');
  evalOqiDetails.hidden = false;
}

function renderEvaluation(data) {
  evalPanel.hidden = false;
  evalError.hidden = true;

  renderFramework(data.framework);
  renderSystemVerdict(data.system_verdict);
  renderSystemMetrics(data.system_metrics);
  renderChecklist(data.system_checklist);
  renderOqi(data.oqi);

  evalDetails.hidden = false;
  evalJson.textContent = JSON.stringify(data, null, 2);
}

async function runEvaluation(jobId, { silent = false } = {}) {
  if (!evaluateBtn && !silent) return null;
  evalError.hidden = true;
  if (evaluateBtn) {
    evaluateBtn.disabled = true;
    evaluateBtn.textContent = 'Evaluating…';
  }

  try {
    const res = await fetch(`${API}/api/jobs/${jobId}/evaluate`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || data.message || 'Evaluation failed');
    }
    renderEvaluation(data);
    showEvaluateButton(jobId, { alreadyLoaded: true });
    return data;
  } catch (err) {
    evalError.textContent = err.message || String(err);
    evalError.hidden = false;
    showEvaluateButton(jobId);
    return null;
  } finally {
    if (evaluateBtn) {
      evaluateBtn.disabled = false;
      if (!evaluateBtn.hidden) {
        evaluateBtn.textContent = 'Re-run evaluation';
      }
    }
  }
}
