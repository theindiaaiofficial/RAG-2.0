const $ = (id) => document.getElementById(id);
let currentSession = null;
let busy = false;

function toast(message) {
  const el = $('toast'); el.textContent = message; el.classList.add('show');
  window.setTimeout(() => el.classList.remove('show'), 2600);
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function setBusy(value) {
  busy = value; $('send-button').disabled = value; $('message-input').disabled = value;
}
function scrollBottom() { $('chat-scroll').scrollTop = $('chat-scroll').scrollHeight; }
function renderMessage(role, content, citations = []) {
  $('welcome').style.display = 'none';
  const wrapper = document.createElement('div'); wrapper.className = `message ${role}`;
  const avatar = document.createElement('div'); avatar.className = 'avatar'; avatar.textContent = role === 'user' ? 'Y' : 'O';
  const bubble = document.createElement('div'); bubble.className = 'bubble';
  bubble.textContent = content;
  wrapper.append(avatar, bubble);
  if (citations?.length) {
    const citeWrap = document.createElement('div'); citeWrap.className = 'citations';
    citations.forEach((citation, index) => {
      const btn = document.createElement('button'); btn.className = 'citation';
      btn.textContent = `${citation.source_id || `Source ${index+1}`} · ${citation.filename}${citation.page ? ` · p.${citation.page}` : ''}`;
      btn.addEventListener('click', () => {
        const detail = wrapper.querySelector('.citation-detail');
        if (detail) detail.remove();
        const panel = document.createElement('div'); panel.className = 'citation-detail';
        panel.textContent = citation.excerpt || citation.title || citation.filename;
        citeWrap.after(panel);
      });
      citeWrap.append(btn);
    });
    bubble.append(citeWrap);
  }
  $('messages').append(wrapper); scrollBottom();
}
function showTyping() {
  const node = document.createElement('div'); node.className = 'message assistant'; node.id = 'typing-indicator';
  node.innerHTML = '<div class="avatar">O</div><div class="typing">Orbit is thinking</div>';
  $('messages').append(node); scrollBottom();
}
async function api(url, options = {}) {
  const response = await fetch(url, {headers:{'Content-Type':'application/json', ...(options.headers || {})}, ...options});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}
async function loadConversations() {
  try {
    const items = await api('/api/conversations');
    $('conversation-list').innerHTML = '';
    items.forEach(item => {
      const el = document.createElement('div'); el.className = 'conversation-item' + (item.session_id === currentSession ? ' active' : '');
      el.textContent = item.title || 'New conversation'; el.title = item.title || '';
      el.addEventListener('click', () => openConversation(item.session_id));
      $('conversation-list').append(el);
    });
  } catch (error) { console.warn('Conversation list unavailable', error); }
}
async function newConversation() {
  try {
    const item = await api('/api/conversations', {method:'POST', body:'{}'});
    currentSession = item.session_id; $('messages').innerHTML = ''; $('welcome').style.display = '';
    $('session-label').textContent = 'New conversation'; $('message-input').focus(); await loadConversations();
    $('sidebar').classList.remove('open');
  } catch (error) { toast(error.message); }
}
async function openConversation(id) {
  try {
    const data = await api(`/api/conversations/${encodeURIComponent(id)}`);
    currentSession = id; $('messages').innerHTML = ''; $('welcome').style.display = data.messages.length ? 'none' : '';
    $('session-label').textContent = data.title;
    data.messages.forEach(item => renderMessage(item.role, item.content, item.citations));
    await loadConversations(); $('sidebar').classList.remove('open');
  } catch (error) { toast(error.message); }
}
async function sendMessage(text) {
  if (busy || !text.trim()) return;
  if (!currentSession) {
    const item = await api('/api/conversations', {method:'POST', body:'{}'}); currentSession = item.session_id;
  }
  $('message-input').value = ''; $('message-input').style.height = 'auto';
  renderMessage('user', text.trim()); setBusy(true); showTyping();
  try {
    const result = await api('/api/chat', {method:'POST', body:JSON.stringify({message:text.trim(), session_id:currentSession})});
    $('typing-indicator')?.remove(); renderMessage('assistant', result.answer, result.citations);
    $('session-label').textContent = 'Conversation'; await loadConversations();
  } catch (error) {
    $('typing-indicator')?.remove(); renderMessage('assistant', `I couldn't complete that request: ${error.message}`);
  } finally { setBusy(false); $('message-input').focus(); }
}
async function refreshStatus() {
  try {
    const [knowledge, readiness] = await Promise.all([
      api('/api/knowledge/status'), fetch('/ready').then(async r => ({ok:r.ok, data:await r.json()}))
    ]);
    $('knowledge-summary').textContent = knowledge.message;
    $('knowledge-dot').className = 'status-dot ' + (knowledge.errors.length ? 'bad' : knowledge.empty ? '' : 'good');
    $('ready-label').textContent = readiness.ok ? 'System ready' : 'Setup required';
    $('ready-dot').className = 'status-dot ' + (readiness.ok ? 'good' : 'bad');
  } catch (error) {
    $('ready-label').textContent = 'Status unavailable'; $('ready-dot').className = 'status-dot bad';
    $('knowledge-summary').textContent = 'Could not read knowledge status.';
  }
}
$('chat-form').addEventListener('submit', e => { e.preventDefault(); sendMessage($('message-input').value); });
$('message-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('chat-form').requestSubmit(); }
});
$('message-input').addEventListener('input', e => { e.target.style.height = 'auto'; e.target.style.height = `${Math.min(e.target.scrollHeight,160)}px`; });
$('new-chat').addEventListener('click', newConversation);
$('mobile-menu').addEventListener('click', () => $('sidebar').classList.toggle('open'));
$('reindex-button').addEventListener('click', async () => {
  $('reindex-button').disabled = true; $('reindex-button').textContent = 'Indexing…';
  try { const result = await api('/api/knowledge/reindex', {method:'POST', body:JSON.stringify({full:false})}); toast(`Indexed ${result.processed_files} file(s); ${result.chunk_count} chunks available.`); }
  catch (error) { toast(error.message); }
  finally { $('reindex-button').disabled = false; $('reindex-button').textContent = '↻ Reindex knowledge'; await refreshStatus(); }
});
document.querySelectorAll('.suggestion').forEach(button => button.addEventListener('click', () => sendMessage(button.dataset.prompt)));
(async function init() { await loadConversations(); await refreshStatus(); })();
