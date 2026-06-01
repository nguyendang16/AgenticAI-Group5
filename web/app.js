const API = '';

const uploadForm = document.getElementById('upload-form');
const pdfFile = document.getElementById('pdf-file');
const fileName = document.getElementById('file-name');
const submitBtn = document.getElementById('submit-btn');
const submitError = document.getElementById('submit-error');
const progressPanel = document.getElementById('progress-panel');
const resultPanel = document.getElementById('result-panel');
const statusBadge = document.getElementById('status-badge');
const statusMessage = document.getElementById('status-message');
const annotationCount = document.getElementById('annotation-count');
const toolCalls = document.getElementById('tool-calls');
const toolTimeline = document.getElementById('tool-timeline');
const reportMarkdown = document.getElementById('report-markdown');
const reportPdf = document.getElementById('report-pdf');
const downloadPdf = document.getElementById('download-pdf');
const resultError = document.getElementById('result-error');
const tabMd = document.getElementById('tab-md');
const tabPdf = document.getElementById('tab-pdf');

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
  resultPanel.hidden = true;
  resultError.hidden = true;

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
      await loadResult(jobId);
    } else if (status.status === 'failed') {
      clearInterval(pollTimer);
      pollTimer = null;
      submitError.textContent = status.error || status.message || 'Job failed';
      submitError.hidden = false;
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

async function loadResult(jobId) {
  resultPanel.hidden = false;
  resultError.hidden = true;

  try {
    const mdRes = await fetch(`${API}/api/jobs/${jobId}/report.md`);
    if (!mdRes.ok) throw new Error('Markdown report not available');
    const md = await mdRes.text();
    reportMarkdown.textContent = md;

    const pdfUrl = `${API}/api/jobs/${jobId}/report.pdf`;
    reportPdf.src = pdfUrl;
    downloadPdf.href = pdfUrl;
    downloadPdf.hidden = false;
  } catch (err) {
    resultError.textContent = err.message || String(err);
    resultError.hidden = false;
  }
}

tabMd.addEventListener('click', () => {
  tabMd.classList.add('active');
  tabPdf.classList.remove('active');
  reportMarkdown.hidden = false;
  reportPdf.hidden = true;
});

tabPdf.addEventListener('click', () => {
  tabPdf.classList.add('active');
  tabMd.classList.remove('active');
  reportMarkdown.hidden = true;
  reportPdf.hidden = false;
});
