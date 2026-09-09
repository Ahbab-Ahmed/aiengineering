/* app.js — Frontend logic for the Medical Guideline Assistant */

const API = '';   // same origin
let currentSessionId = null;
let isStreaming = false;

// ── DOM refs ──────────────────────────────────────────────────────
const messagesEl    = document.getElementById('messages');
const welcomeEl     = document.getElementById('welcome');
const inputEl       = document.getElementById('query-input');
const sendBtn       = document.getElementById('send-btn');
const sessionsList  = document.getElementById('sessions-list');
const sessionTitle  = document.getElementById('session-title');
const newSessionBtn = document.getElementById('new-session-btn');
const notesBtn      = document.getElementById('notes-btn');
const notesPanel    = document.getElementById('notes-panel');
const notesClose    = document.getElementById('notes-close');
const notesContent  = document.getElementById('notes-content');
const notesSave     = document.getElementById('notes-save');

// ── Init ──────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  await loadSessions();

  // Auto-resize textarea
  inputEl.addEventListener('input', () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
    sendBtn.disabled = inputEl.value.trim() === '' || isStreaming;
  });

  inputEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!sendBtn.disabled) sendMessage();
    }
  });

  sendBtn.addEventListener('click', sendMessage);
  newSessionBtn.addEventListener('click', createSession);

  notesBtn.addEventListener('click', toggleNotes);
  notesClose.addEventListener('click', () => notesPanel.classList.add('hidden'));
  notesSave.addEventListener('click', saveNotes);
});

// ── Sessions ──────────────────────────────────────────────────────
async function loadSessions() {
  const res  = await fetch(`${API}/sessions`);
  const list = await res.json();
  renderSessions(list);
  if (list.length > 0 && !currentSessionId) {
    await selectSession(list[0].id);
  }
}

function renderSessions(list) {
  sessionsList.innerHTML = '';
  list.forEach(s => {
    const el = document.createElement('div');
    el.className = 'session-item' + (s.id === currentSessionId ? ' active' : '');
    el.innerHTML = `<span class="session-icon">💬</span><span class="session-name">${escHtml(s.name)}</span>`;
    el.addEventListener('click', () => selectSession(s.id));
    sessionsList.appendChild(el);
  });
}

async function createSession() {
  const res = await fetch(`${API}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: `Session ${new Date().toLocaleString('en-GB', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}` }),
  });
  const s = await res.json();
  await loadSessions();
  await selectSession(s.id);
}

async function selectSession(id) {
  currentSessionId = id;
  const res = await fetch(`${API}/sessions/${id}`);
  const s   = await res.json();

  sessionTitle.textContent = s.name;
  sendBtn.disabled = inputEl.value.trim() === '';

  // Render existing messages
  messagesEl.innerHTML = '';
  if (s.messages.length === 0) {
    messagesEl.appendChild(welcomeEl);
    welcomeEl.style.display = 'block';
  } else {
    welcomeEl.style.display = 'none';
    s.messages.forEach((m, i) => renderMessage(m.role, m.content, i, []));
  }

  scrollToBottom();
  await loadSessions();   // refresh sidebar active state
  await loadNotes();
}

// ── Messaging ─────────────────────────────────────────────────────
async function sendMessage() {
  if (isStreaming || !currentSessionId) return;
  const query = inputEl.value.trim();
  if (!query) return;

  isStreaming = true;
  sendBtn.disabled = true;
  inputEl.value = '';
  inputEl.style.height = 'auto';

  // Hide welcome
  welcomeEl.style.display = 'none';

  // Show user message
  renderMessage('user', query, null, []);
  scrollToBottom();

  // Typing indicator
  const typingId = 'typing-' + Date.now();
  messagesEl.insertAdjacentHTML('beforeend', `
    <div class="message assistant" id="${typingId}">
      <div class="avatar">🏥</div>
      <div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>
    </div>
  `);
  scrollToBottom();

  try {
    const res = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: currentSessionId, message: query }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    let fullText     = '';
    let citations    = [];
    let isRefusal    = false;
    let overrideText = '';

    // Replace typing indicator with empty bubble
    const typingEl = document.getElementById(typingId);
    const msgEl = document.createElement('div');
    msgEl.className = 'message assistant';
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = '🏥';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    msgEl.appendChild(avatar);
    msgEl.appendChild(bubble);
    typingEl.replaceWith(msgEl);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const data = JSON.parse(line.slice(6).trim());

          if (data.type === 'token') {
            fullText += data.content;
            bubble.innerHTML = renderMarkdown(fullText);
            scrollToBottom();

          } else if (data.type === 'guardrail') {
            isRefusal = true;
            bubble.classList.add('refusal');

          } else if (data.type === 'override') {
            overrideText = data.content;
            isRefusal = true;
            bubble.classList.add('refusal');
            bubble.innerHTML = renderMarkdown(overrideText);

          } else if (data.type === 'citations') {
            citations = data.citations || [];

          } else if (data.type === 'done') {
            // Finalise the message
            finaliseBubble(bubble, fullText || overrideText, citations, isRefusal);
          }
        } catch (_) {}
      }
    }
  } catch (e) {
    const typingEl = document.getElementById(typingId);
    if (typingEl) {
      typingEl.querySelector('.bubble').textContent = 'Connection error. Please try again.';
    }
  }

  isStreaming = false;
  sendBtn.disabled = inputEl.value.trim() === '';
  scrollToBottom();
}

function finaliseBubble(bubble, text, citations, isRefusal) {
  // Separate disclaimer from main answer for styling
  const disclaimerMark = '\n\n---\n';
  const disclaimerIdx = text.indexOf(disclaimerMark);
  let mainText = text;
  let disclaimerText = '';
  if (disclaimerIdx > -1) {
    mainText = text.slice(0, disclaimerIdx);
    disclaimerText = text.slice(disclaimerIdx + disclaimerMark.length);
  }

  let html = renderMarkdown(mainText);

  // Citations
  if (citations.length > 0 && !isRefusal) {
    html += '<div class="citations">';
    citations.forEach(c => {
      const label = `${c.source || 'Source'} p.${c.page || '?'}`;
      html += `<span class="citation-chip" title="${escHtml(c.chunk_id || '')}">${escHtml(label)}</span>`;
    });
    html += '</div>';
  }

  // Disclaimer
  if (disclaimerText) {
    html += `<div class="disclaimer-text">${escHtml(disclaimerText.trim())}</div>`;
  }

  bubble.innerHTML = html;

  // Add feedback buttons
  if (!isRefusal) {
    const idx = document.querySelectorAll('.message.assistant').length - 1;
    bubble.insertAdjacentHTML('afterend', `
      <div class="feedback-row">
        <button class="feedback-btn" onclick="sendFeedback(this, ${idx}, 1)">👍</button>
        <button class="feedback-btn" onclick="sendFeedback(this, ${idx}, -1)">👎</button>
      </div>
    `);
  }
}

// ── Feedback ──────────────────────────────────────────────────────
async function sendFeedback(btn, idx, score) {
  if (!currentSessionId) return;
  await fetch(`${API}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: currentSessionId, message_index: idx, score }),
  });
  btn.parentElement.querySelectorAll('.feedback-btn').forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');
}

// ── Notes ─────────────────────────────────────────────────────────
function toggleNotes() {
  notesPanel.classList.toggle('hidden');
}
async function loadNotes() {
  if (!currentSessionId) return;
  const res = await fetch(`${API}/notes/${currentSessionId}`);
  const d   = await res.json();
  notesContent.value = d.content || '';
}
async function saveNotes() {
  if (!currentSessionId) return;
  await fetch(`${API}/notes/${currentSessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content: notesContent.value }),
  });
  notesSave.textContent = '✅ Saved';
  setTimeout(() => { notesSave.textContent = 'Save Notes'; }, 1500);
}

// ── Helpers ───────────────────────────────────────────────────────
function renderMessage(role, content, idx, citations) {
  const isUser = role === 'user';
  const el = document.createElement('div');
  el.className = `message ${role}`;
  el.innerHTML = `
    <div class="avatar">${isUser ? '👤' : '🏥'}</div>
    <div class="bubble">${isUser ? escHtml(content) : renderMarkdown(content)}</div>
  `;
  messagesEl.appendChild(el);
}

function scrollToBottom() {
  const container = document.getElementById('messages-container');
  container.scrollTop = container.scrollHeight;
}

function setQuery(text) {
  inputEl.value = text;
  inputEl.dispatchEvent(new Event('input'));
  inputEl.focus();
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderMarkdown(text) {
  // Simple markdown renderer (no external dependency)
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm,  '<h2>$1</h2>')
    .replace(/^# (.+)$/gm,   '<h1>$1</h1>')
    .replace(/^\- (.+)$/gm,  '<li>$1</li>')
    .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/^(?!<[hul])(.+)$/gm, '<p>$1</p>')
    .replace(/\[chunk_([a-f0-9]+)\]/g, '<span class="citation-chip" style="cursor:default">[chunk_$1]</span>');
}
