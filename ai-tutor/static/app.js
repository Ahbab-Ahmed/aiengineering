/* ── State ─────────────────────────────────────────────────── */
let currentSessionId = null;
let currentMode      = 'learn';
let isStreaming      = false;
let curriculum       = [];
let messageCount     = 0;   // tracks # of assistant messages for feedback indexing
let notesSaveTimer   = null;
let notesOpen        = false;

/* ── Marked config ─────────────────────────────────────────── */
marked.setOptions({ breaks: true, gfm: true });

/* ── Init ──────────────────────────────────────────────────── */
async function init() {
  await waitForReady();
  await loadCurriculum();
  await loadSessions();
  setupEventListeners();
}

async function waitForReady() {
  const sub = document.getElementById('loading-sub');
  while (true) {
    try {
      const r = await fetch('/health');
      const data = await r.json();
      if (data.ready) break;
      if (data.error) { sub.textContent = '❌ ' + data.error; return; }
    } catch (_) {}
    await sleep(1500);
  }
  document.getElementById('loading-overlay').style.display = 'none';
  document.getElementById('app').style.display = 'grid';
}

/* ── Sessions ──────────────────────────────────────────────── */
async function loadSessions() {
  const res = await fetch('/sessions');
  const list = await res.json();
  renderSessionList(list);
  if (list.length > 0) await selectSession(list[0].id);
  else await createNewSession();
}

function renderSessionList(list) {
  document.getElementById('sessions-list').innerHTML = list.map(s => `
    <div class="session-item ${s.id === currentSessionId ? 'active' : ''}"
         data-id="${s.id}" onclick="selectSession('${s.id}')">
      <div class="session-item-info">
        <div class="session-item-name">${escHtml(s.name)}</div>
        <div class="session-item-date">${formatDate(s.updated_at)}</div>
      </div>
      <button class="session-delete" onclick="deleteSession(event,'${s.id}')" title="Delete">×</button>
    </div>
  `).join('');
}

async function createNewSession() {
  const res = await fetch('/sessions', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: null }),
  });
  const s = await res.json();
  await selectSession(s.id);
  await refreshSessionList();
}

async function selectSession(id) {
  currentSessionId = id;
  messageCount = 0;
  const res = await fetch(`/sessions/${id}`);
  const session = await res.json();
  renderMessages(session.messages);
  document.querySelectorAll('.session-item').forEach(el =>
    el.classList.toggle('active', el.dataset.id === id)
  );
  enableInput();
  // Load notes for this session
  await loadNotes(id);
}

async function deleteSession(e, id) {
  e.stopPropagation();
  await fetch(`/sessions/${id}`, { method: 'DELETE' });
  if (id === currentSessionId) { currentSessionId = null; clearMessages(); }
  await refreshSessionList();
  if (!currentSessionId) await createNewSession();
}

async function refreshSessionList() {
  const res = await fetch('/sessions');
  renderSessionList(await res.json());
}

/* ── Curriculum ────────────────────────────────────────────── */
async function loadCurriculum() {
  const res = await fetch('/curriculum');
  curriculum = await res.json();
  renderCurriculumList();
}

function getCurriculumProgress() {
  try { return JSON.parse(localStorage.getItem('curriculum_progress') || '[]'); }
  catch { return []; }
}

function markCurriculumSessionComplete(sessionId) {
  const done = getCurriculumProgress();
  if (!done.includes(sessionId)) {
    done.push(sessionId);
    localStorage.setItem('curriculum_progress', JSON.stringify(done));
    renderCurriculumList();
  }
}

function renderCurriculumList() {
  const done = getCurriculumProgress();
  document.getElementById('curriculum-list').innerHTML = curriculum.map(s => {
    const completed = done.includes(s.id);
    return `
      <div class="curriculum-item ${completed ? 'completed' : ''}"
           onclick="startCurriculumSession(${s.id})"
           title="${escHtml(s.description)}">
        <div class="curriculum-num">${completed ? '✓' : s.id}</div>
        <div class="curriculum-item-title">${escHtml(s.title)}</div>
      </div>
    `;
  }).join('');
}

async function startCurriculumSession(sessionNum) {
  const session = curriculum.find(s => s.id === sessionNum);
  if (!session) return;
  setMode('curriculum');
  const msg = `Start curriculum session ${sessionNum}: "${session.title}". Topics: ${session.topics.join(', ')}.`;
  await sendMessageText(msg);
}

/* ── Messages rendering ────────────────────────────────────── */
function renderMessages(messages) {
  const container = document.getElementById('messages');
  container.innerHTML = '';
  messageCount = 0;

  if (!messages || messages.length === 0) {
    container.innerHTML = welcomeHTML();
    bindSuggestionChips();
    return;
  }

  for (const msg of messages) {
    if (msg.role === 'assistant') {
      const idx = messageCount++;
      appendMessage(msg.role, msg.content, [], false, idx);
    } else {
      appendMessage(msg.role, msg.content, [], false, null);
    }
  }
  scrollToBottom();
}

function welcomeHTML() {
  return `
    <div class="welcome-msg" id="welcome-msg">
      <div class="welcome-icon">🎓</div>
      <h2>Welcome to your AI Engineering Tutor</h2>
      <p>Ask anything about Chip Huyen's <em>AI Engineering</em> book. I'll teach with structured explanations, real-world examples, and precise page + chapter references.</p>
      <div class="welcome-suggestions">
        <button class="suggestion-chip" data-msg="What is AI Engineering and how is it different from traditional ML engineering?">What is AI Engineering?</button>
        <button class="suggestion-chip" data-msg="Explain RAG in depth — how does it work and when should I use it?">Explain RAG</button>
        <button class="suggestion-chip" data-msg="When should I finetune vs use RAG? Give me a decision framework.">Finetuning vs RAG</button>
        <button class="suggestion-chip" data-msg="What are AI Agents and how do they work with tools?">AI Agents</button>
        <button class="suggestion-chip" data-msg="How do I evaluate an AI application? What metrics and methods should I use?">Evaluation</button>
        <button class="suggestion-chip" data-msg="Quiz me on Prompt Engineering">🧠 Quiz me</button>
      </div>
    </div>`;
}

function clearMessages() {
  document.getElementById('messages').innerHTML = welcomeHTML();
  messageCount = 0;
  bindSuggestionChips();
}

function appendMessage(role, content, sources = [], streaming = false, feedbackIndex = null) {
  const welcome = document.getElementById('welcome-msg');
  if (welcome) welcome.remove();

  const container = document.getElementById('messages');
  const id = 'msg-' + Date.now() + '-' + Math.random().toString(36).slice(2, 5);

  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.id = id;

  const avatar = role === 'user' ? '👤' : '🤖';
  const bubbleContent = role === 'user'
    ? `<p>${escHtml(content)}</p>`
    : (streaming ? '' : renderMarkdown(content));

  div.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-content">
      <div class="msg-bubble">${bubbleContent}${streaming ? '<span class="streaming-cursor"></span>' : ''}</div>
      ${!streaming && sources.length > 0 ? renderSources(sources) : ''}
      ${!streaming && role === 'assistant' && feedbackIndex !== null ? renderFeedback(feedbackIndex) : ''}
    </div>
  `;

  container.appendChild(div);

  if (!streaming && role === 'assistant') {
    div.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));
  }

  scrollToBottom();
  return id;
}

function renderMarkdown(text) {
  return marked.parse(text || '');
}

function renderSources(sources) {
  if (!sources || sources.length === 0) return '';
  const chips = sources.map(s => {
    const hasChapter = s.chapter_title && s.chapter_num;
    const label = hasChapter
      ? `📖 Ch.${s.chapter_num} p.${s.page}`
      : `📖 p.${s.page}`;
    const title = hasChapter ? escHtml(s.chapter_title) : `Page ${s.page}`;
    return `<span class="source-chip ${hasChapter ? 'has-chapter' : ''}" title="${title}">${label}</span>`;
  }).join('');
  return `<div class="msg-sources">${chips}</div>`;
}

function renderFeedback(idx) {
  return `
    <div class="msg-feedback" id="feedback-${idx}">
      <span class="feedback-label">Helpful?</span>
      <button class="feedback-btn" onclick="sendFeedback(${idx}, 1, this)" title="👍 Helpful">👍</button>
      <button class="feedback-btn" onclick="sendFeedback(${idx}, -1, this)" title="👎 Not helpful">👎</button>
    </div>`;
}

async function sendFeedback(messageIndex, score, btn) {
  if (!currentSessionId) return;
  await fetch('/feedback', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: currentSessionId, message_index: messageIndex, score }),
  });
  // Update button states
  const container = document.getElementById(`feedback-${messageIndex}`);
  if (!container) return;
  container.querySelectorAll('.feedback-btn').forEach(b => {
    b.classList.remove('selected-up', 'selected-down');
  });
  btn.classList.add(score === 1 ? 'selected-up' : 'selected-down');
}

/* ── Streaming ─────────────────────────────────────────────── */
async function sendMessage() {
  const input = document.getElementById('user-input');
  const text = input.value.trim();
  if (!text || isStreaming || !currentSessionId) return;
  input.value = '';
  autoResizeTextarea(input);
  await sendMessageText(text);
}

async function sendMessageText(text) {
  if (isStreaming || !currentSessionId) return;

  appendMessage('user', text);
  const assistantMsgId = appendMessage('assistant', '', [], true, null);
  const thisIndex = messageCount++;

  setStreaming(true);
  let fullText = '';
  let buffer = '';
  let lastSources = [];

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: currentSessionId, message: text, mode: currentMode }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split('\n\n');
      buffer = events.pop() || '';

      for (const event of events) {
        for (const line of event.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'token') {
              fullText += data.content;
              updateStreamingBubble(assistantMsgId, fullText);
            } else if (data.type === 'sources') {
              lastSources = data.content || [];
            } else if (data.type === 'error') {
              fullText = `❌ Error: ${data.content}`;
              updateStreamingBubble(assistantMsgId, fullText);
            }
          } catch (_) {}
        }
      }
    }

    // Finalize: full markdown, sources, feedback
    finalizeAssistantMessage(assistantMsgId, fullText, lastSources, thisIndex);

  } catch (e) {
    finalizeAssistantMessage(assistantMsgId, `❌ Error: ${e.message}`, [], thisIndex);
  } finally {
    setStreaming(false);
    await refreshSessionList();
  }
}

/* Progressive markdown rendering — re-renders every token */
function updateStreamingBubble(msgId, text) {
  const el = document.getElementById(msgId);
  if (!el) return;
  const bubble = el.querySelector('.msg-bubble');
  if (!bubble) return;
  bubble.innerHTML = renderMarkdown(text) + '<span class="streaming-cursor"></span>';
  scrollToBottom();
}

function finalizeAssistantMessage(msgId, text, sources, feedbackIndex) {
  const el = document.getElementById(msgId);
  if (!el) return;
  const bubble = el.querySelector('.msg-bubble');
  if (bubble) {
    bubble.innerHTML = renderMarkdown(text);
    bubble.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));
  }
  const content = el.querySelector('.msg-content');
  if (content) {
    // Remove any existing sources/feedback
    content.querySelectorAll('.msg-sources, .msg-feedback').forEach(e => e.remove());
    if (sources.length > 0) content.insertAdjacentHTML('beforeend', renderSources(sources));
    content.insertAdjacentHTML('beforeend', renderFeedback(feedbackIndex));
  }
  scrollToBottom();
}

/* ── Mode ──────────────────────────────────────────────────── */
function setMode(mode) {
  currentMode = mode;
  document.querySelectorAll('.mode-tab').forEach(t =>
    t.classList.toggle('active', t.dataset.mode === mode)
  );
  const dot = document.getElementById('mode-dot');
  dot.className = 'mode-dot';
  dot.classList.add({ learn: 'learn-dot', quiz: 'quiz-dot', curriculum: 'curr-dot' }[mode] || 'learn-dot');

  document.getElementById('mode-label').textContent =
    { learn: 'Learn Mode', quiz: 'Quiz Mode', curriculum: 'Curriculum Mode' }[mode] || 'Learn Mode';

  document.getElementById('user-input').placeholder =
    { learn: 'Ask anything about AI Engineering…', quiz: 'Type a topic to quiz yourself on (e.g. "RAG", "Finetuning")…', curriculum: 'Ask about a session or click one in the sidebar…' }[mode] || '';
}

/* ── Notes Panel ───────────────────────────────────────────── */
async function loadNotes(sessionId) {
  try {
    const res = await fetch(`/notes/${sessionId}`);
    const data = await res.json();
    document.getElementById('notes-textarea').value = data.content || '';
    setNotesStatus('Auto-saved', 'saved');
  } catch (_) {}
}

function toggleNotes() {
  notesOpen = !notesOpen;
  document.getElementById('notes-panel').classList.toggle('open', notesOpen);
  document.getElementById('notes-toggle').classList.toggle('active', notesOpen);
  document.querySelector('.main').classList.toggle('notes-open', notesOpen);
}

function scheduleNotesSave() {
  setNotesStatus('Saving…', 'saving');
  clearTimeout(notesSaveTimer);
  notesSaveTimer = setTimeout(async () => {
    if (!currentSessionId) return;
    const content = document.getElementById('notes-textarea').value;
    try {
      await fetch(`/notes/${currentSessionId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
      });
      setNotesStatus('Auto-saved', 'saved');
    } catch (_) {
      setNotesStatus('Save failed', '');
    }
  }, 1500);
}

function setNotesStatus(text, cls) {
  const el = document.getElementById('notes-status');
  el.textContent = text;
  el.className = 'notes-status' + (cls ? ` ${cls}` : '');
}

/* ── UI helpers ────────────────────────────────────────────── */
function setStreaming(val) {
  isStreaming = val;
  document.getElementById('send-btn').disabled = val;
  document.getElementById('user-input').disabled = val;
}

function enableInput() {
  document.getElementById('send-btn').disabled = false;
  document.getElementById('user-input').disabled = false;
}

function scrollToBottom() {
  const c = document.getElementById('messages');
  c.scrollTop = c.scrollHeight;
}

function autoResizeTextarea(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function formatDate(iso) {
  if (!iso) return '';
  try {
    return new Date(iso + 'Z').toLocaleDateString('en-US',
      { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  } catch { return ''; }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function bindSuggestionChips() {
  document.querySelectorAll('.suggestion-chip').forEach(chip =>
    chip.addEventListener('click', () => sendMessageText(chip.dataset.msg))
  );
}

/* ── Event listeners ───────────────────────────────────────── */
function setupEventListeners() {
  document.getElementById('send-btn').addEventListener('click', sendMessage);

  document.getElementById('user-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  document.getElementById('user-input').addEventListener('input', function () {
    autoResizeTextarea(this);
    document.getElementById('send-btn').disabled = !this.value.trim() || isStreaming;
  });

  document.querySelectorAll('.mode-tab').forEach(tab =>
    tab.addEventListener('click', () => setMode(tab.dataset.mode))
  );

  document.getElementById('new-session-btn').addEventListener('click', createNewSession);
  document.getElementById('notes-toggle').addEventListener('click', toggleNotes);
  document.getElementById('notes-close').addEventListener('click', toggleNotes);
  document.getElementById('notes-textarea').addEventListener('input', scheduleNotesSave);

  bindSuggestionChips();
}

/* ── Boot ──────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', init);
