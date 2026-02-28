/* ═══════════════════════════════════════════════════════════
   IRONCLIW — app.js : Full UI Logic
   Boot Sequence · Particle Engine · Waveform · WebSocket
   Holographic Orb · Mic Input · Chat · Action Log
   ═══════════════════════════════════════════════════════════ */

'use strict';

// ─────────────────────────────────────────────
// 0. ELECTRON IPC (safe — only available in Electron)
// ─────────────────────────────────────────────
const electronAPI = window.electronAPI || null;

if (electronAPI) {
    document.getElementById('btn-minimize').addEventListener('click', () => electronAPI.minimize());
    document.getElementById('btn-maximize').addEventListener('click', () => electronAPI.maximize());
    document.getElementById('btn-close').addEventListener('click', () => electronAPI.close());

    // Sync maximize icon
    electronAPI.onWindowMaximized((isMax) => {
        document.getElementById('btn-maximize').textContent = isMax ? '❐' : '▢';
    });

    // ── Backend Status from Main Process ──────────────────────────────────
    electronAPI.onBackendStatus((status) => {
        const backendEl = document.getElementById('stat-backend');
        if (status.running) {
            if (backendEl) {
                backendEl.textContent = 'ONLINE';
                backendEl.className = 'stat-val online';
            }
            addAction('✅ Backend process ready (PID: ' + (status.pid || '?') + ')');
            showToast('Backend process started — connecting...', 'success');
        } else {
            if (backendEl) {
                backendEl.textContent = 'OFFLINE';
                backendEl.className = 'stat-val';
            }
            if (status.error) {
                addAction('⚠ Backend error: ' + status.error);
                showToast('Backend failed to start: ' + status.error, 'error');
            }
        }
    });

    // ── Backend Log Stream → Action Log ────────────────────────────────────
    electronAPI.onBackendLog((line) => {
        // Only show important log lines (not every debug line)
        if (line && (line.includes('ERROR') || line.includes('WARNING') ||
            line.includes('started') || line.includes('ready') ||
            line.includes('WebSocket') || line.includes('Ironcliw'))) {
            addAction('🖥 ' + line.substring(0, 80));
        }
    });
}

// ─────────────────────────────────────────────
// 1. BOOT SEQUENCE
// ─────────────────────────────────────────────
const BOOT_STEPS = [
    { msg: 'Loading cortex modules...', pct: 8 },
    { msg: 'Launching backend engine...', pct: 18 },
    { msg: 'Initializing neural mesh...', pct: 30 },
    { msg: 'Calibrating voice pipelines...', pct: 44 },
    { msg: 'Mounting holographic renderer...', pct: 56 },
    { msg: 'Connecting WebSocket bridge...', pct: 68 },
    { msg: 'Activating autonomous agents...', pct: 82 },
    { msg: 'Syncing knowledge base...', pct: 92 },
    { msg: 'IRONCLIW ONLINE', pct: 100 },
];

let bootDone = false;

function runBoot() {
    const bootBar = document.getElementById('boot-bar');
    const bootStatus = document.getElementById('boot-status');
    const bootScreen = document.getElementById('boot-screen');
    const app = document.getElementById('app');

    let step = 0;
    function nextStep() {
        if (step >= BOOT_STEPS.length) {
            // Fade out boot screen
            setTimeout(() => {
                bootScreen.style.opacity = '0';
                bootScreen.style.pointerEvents = 'none';
                app.classList.remove('hidden');
                bootDone = true;
                initParticles();
                initWaveform();
                initHoloParticles();
                connectWebSocket();
                startUptimeClock();
                startMetricsSim();
                setTimeout(() => {
                    bootScreen.style.display = 'none';
                    showToast('Ironcliw systems fully operational', 'success');
                }, 900);
            }, 400);
            return;
        }
        const s = BOOT_STEPS[step++];
        bootBar.style.width = s.pct + '%';
        bootStatus.textContent = s.msg;
        const delay = s.pct === 100 ? 600 : 200 + Math.random() * 250;
        setTimeout(nextStep, delay);
    }
    nextStep();
}

window.addEventListener('DOMContentLoaded', () => {
    setTimeout(runBoot, 400);
});

// ─────────────────────────────────────────────
// 2. PARTICLE SYSTEM (background)
// ─────────────────────────────────────────────
function initParticles() {
    const canvas = document.getElementById('particle-canvas');
    const ctx = canvas.getContext('2d');

    let W, H, particles = [];

    function resize() {
        W = canvas.width = window.innerWidth;
        H = canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    const COUNT = 90;
    for (let i = 0; i < COUNT; i++) {
        particles.push(makeParticle(W, H));
    }

    function makeParticle(w, h) {
        const hue = Math.random() < 0.6 ? 190 : (Math.random() < 0.5 ? 220 : 270);
        return {
            x: Math.random() * w,
            y: Math.random() * h,
            r: 0.4 + Math.random() * 1.2,
            vx: (Math.random() - 0.5) * 0.25,
            vy: (Math.random() - 0.5) * 0.25,
            alpha: 0.2 + Math.random() * 0.5,
            hue,
        };
    }

    function draw() {
        ctx.clearRect(0, 0, W, H);
        particles.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${p.hue}, 100%, 70%, ${p.alpha})`;
            ctx.fill();

            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0) p.x = W;
            if (p.x > W) p.x = 0;
            if (p.y < 0) p.y = H;
            if (p.y > H) p.y = 0;
        });
        requestAnimationFrame(draw);
    }
    draw();
}

// ─────────────────────────────────────────────
// 3. WAVEFORM VISUALIZER (bottom bar)
// ─────────────────────────────────────────────
let waveActive = false;
let waveContext = null;

function initWaveform() {
    const canvas = document.getElementById('waveform-canvas');
    const ctx = canvas.getContext('2d');
    waveContext = ctx;

    let W, H;
    function resize() {
        W = canvas.width = window.innerWidth;
        H = canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    let phase = 0;
    let amp = 0;          // 0 = silent, increases when active
    let targetAmp = 0;

    window._setWaveAmp = (v) => { targetAmp = v; };  // external hook

    function draw() {
        ctx.clearRect(0, 0, W, H);

        // Smoothly track target amplitude
        amp += (targetAmp - amp) * 0.08;
        if (amp < 0.005) { requestAnimationFrame(draw); return; }

        const baseY = H - 60;
        const barCount = 80;
        const barW = W / barCount;
        const grad = ctx.createLinearGradient(0, baseY - 60, 0, baseY);
        grad.addColorStop(0, 'rgba(0, 245, 255, 0.7)');
        grad.addColorStop(1, 'rgba(0, 100, 255, 0.1)');

        ctx.fillStyle = grad;

        for (let i = 0; i < barCount; i++) {
            const normalizedI = (i / barCount) * Math.PI * 4;
            const h = amp * 50 * (0.3 + 0.7 * Math.abs(Math.sin(normalizedI + phase)));
            ctx.fillRect(i * barW, baseY - h, barW - 1, h);
        }

        phase += 0.08;
        requestAnimationFrame(draw);
    }
    draw();
}

// ─────────────────────────────────────────────
// 4. HOLOGRAPHIC ORB PARTICLES (radius orbit)
// ─────────────────────────────────────────────
function initHoloParticles() {
    const container = document.getElementById('holo-particles');
    const count = 12;
    for (let i = 0; i < count; i++) {
        const el = document.createElement('div');
        el.className = 'holo-particle';

        const angle = (i / count) * 360;
        const radius = 110 + Math.random() * 10;
        const delay = Math.random() * 4;
        const duration = 4 + Math.random() * 3;
        const size = 2 + Math.random() * 2;

        el.style.cssText = `
      width: ${size}px;
      height: ${size}px;
      animation: orbit-particle ${duration}s linear ${delay}s infinite;
      --r: ${radius}px;
      --angle: ${angle}deg;
      left: calc(50% - ${size / 2}px);
      top: calc(50% - ${size / 2}px);
    `;
        container.appendChild(el);
    }

    // Inject keyframe for orbit-particle
    const style = document.createElement('style');
    style.textContent = `
    @keyframes orbit-particle {
      from { transform: rotate(var(--angle)) translateX(var(--r)); opacity: 0.8; }
      50%  { opacity: 0.3; }
      to   { transform: rotate(calc(var(--angle) + 360deg)) translateX(var(--r)); opacity: 0.8; }
    }
  `;
    document.head.appendChild(style);
}

// ─────────────────────────────────────────────
// 5. WEBSOCKET — Ironcliw Python Backend
// ─────────────────────────────────────────────
const WS_URL = 'ws://localhost:8010/ws';
let ws = null;
let wsRetryTimer = null;
let wsConnected = false;
let wsRetryCount = 0;
const WS_MAX_RETRY = 10;
const WS_RETRY_DELAY = 4000;

function connectWebSocket() {
    if (ws) { try { ws.close(); } catch (e) { } }

    try {
        ws = new WebSocket(WS_URL);
    } catch (e) {
        scheduleRetry();
        return;
    }

    ws.onopen = () => {
        wsConnected = true;
        wsRetryCount = 0;
        setConnectionStatus(true);
        showToast('Backend connected — ws://localhost:8010/ws', 'success');
        document.getElementById('stat-backend').textContent = 'ONLINE';
        document.getElementById('stat-backend').className = 'stat-val online';
        addAction('WebSocket connected to backend');
    };

    ws.onclose = () => {
        wsConnected = false;
        setConnectionStatus(false);
        document.getElementById('stat-backend').textContent = 'OFFLINE';
        document.getElementById('stat-backend').className = 'stat-val';
        if (wsRetryCount < WS_MAX_RETRY) scheduleRetry();
        else showToast('Backend offline — retries exhausted', 'error');
    };

    ws.onerror = () => {
        // onclose fires after onerror, handled there
    };

    ws.onmessage = (event) => {
        let msg;
        try { msg = JSON.parse(event.data); }
        catch { msg = { type: 'raw', text: event.data }; }
        handleBackendMessage(msg);
    };
}

function scheduleRetry() {
    wsRetryCount++;
    document.getElementById('conn-label').textContent = `RETRY ${wsRetryCount}/${WS_MAX_RETRY}`;
    wsRetryTimer = setTimeout(connectWebSocket, WS_RETRY_DELAY);
}

function setConnectionStatus(online) {
    const dot = document.getElementById('conn-dot');
    const label = document.getElementById('conn-label');
    dot.className = online ? 'conn-dot online' : 'conn-dot';
    label.textContent = online ? 'CONNECTED' : 'OFFLINE';
}

function sendToBackend(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(payload));
        return true;
    }
    showToast('Backend not connected', 'warn');
    return false;
}

// ─────────────────────────────────────────────
// 6. BACKEND MESSAGE HANDLER
// ─────────────────────────────────────────────
function handleBackendMessage(msg) {
    const type = msg.type || '';

    switch (type) {
        case 'speak':
        case 'response':
        case 'assistant_message': {
            const text = msg.text || msg.message || msg.response || '';
            stopListeningState();
            startSpeakingState();
            addChatMessage('IRONCLIW', text);
            addAction(`Spoke: "${text.substring(0, 50)}${text.length > 50 ? '…' : ''}"`);
            // Simulate wave while speaking
            const words = text.split(' ').length;
            activateWave(Math.min(words / 30, 1) * 0.8 + 0.2, words * 0.25 * 1000);
            break;
        }

        case 'listening':
            startListeningState();
            break;

        case 'thinking':
        case 'processing':
            setJarvisState('THINKING', 'thinking');
            break;

        case 'stop_speaking':
        case 'idle':
            stopSpeakingState();
            break;

        case 'action': {
            const action = msg.action || msg.text || JSON.stringify(msg);
            addAction(action);
            break;
        }

        case 'status': {
            if (msg.mode) document.getElementById('stat-mode').textContent = msg.mode.toUpperCase();
            if (msg.uptime) document.getElementById('stat-uptime').textContent = msg.uptime;
            break;
        }

        case 'error': {
            showToast(msg.message || 'Backend error', 'error');
            addAction('⚠ Error: ' + (msg.message || 'Unknown'));
            break;
        }

        case 'notification': {
            showToast(msg.text || msg.message, 'info');
            break;
        }

        default:
            // Generic message with text → add to chat
            if (msg.text || msg.message) {
                addChatMessage('Ironcliw', msg.text || msg.message);
            }
    }
}

// ─────────────────────────────────────────────
// 7. Ironcliw STATE MACHINE
// ─────────────────────────────────────────────
let currentState = 'standby';
let speakTimer = null;
let listenTimer = null;

function setJarvisState(label, stateClass) {
    currentState = stateClass;
    document.getElementById('state-text').textContent = label;
    const dot = document.getElementById('state-dot');
    dot.className = 'state-dot ' + stateClass;
    document.getElementById('stat-mode').textContent = label;

    // Update orb class
    const core = document.getElementById('holo-core');
    core.className = 'holo-core ' + (stateClass === 'listening' ? 'listening' : stateClass === 'speaking' ? 'speaking' : '');
}

function startListeningState() {
    setJarvisState('LISTENING', 'listening');
    document.getElementById('mic-label').textContent = 'LISTENING...';
    document.getElementById('mic-label').className = 'mic-label active';
    document.getElementById('mic-btn').classList.add('active');
    if (window._setWaveAmp) window._setWaveAmp(0.6);
    clearTimeout(listenTimer);
}

function stopListeningState() {
    document.getElementById('mic-btn').classList.remove('active');
    document.getElementById('mic-label').textContent = 'CLICK TO SPEAK';
    document.getElementById('mic-label').className = 'mic-label';
    if (window._setWaveAmp) window._setWaveAmp(0);
}

function startSpeakingState() {
    setJarvisState('SPEAKING', 'speaking');
}

function stopSpeakingState() {
    setJarvisState('STANDBY', 'active');
    if (window._setWaveAmp) window._setWaveAmp(0);
    clearTimeout(speakTimer);
}

function activateWave(intensity, durationMs) {
    if (window._setWaveAmp) window._setWaveAmp(intensity);
    clearTimeout(speakTimer);
    speakTimer = setTimeout(() => {
        if (window._setWaveAmp) window._setWaveAmp(0);
        setJarvisState('STANDBY', 'active');
    }, durationMs + 500);
}

// ─────────────────────────────────────────────
// 8. CHAT PANEL
// ─────────────────────────────────────────────
const chatLog = document.getElementById('chat-log');

function addChatMessage(sender, text) {
    // Remove typing indicator if present
    const typing = chatLog.querySelector('.typing-entry');
    if (typing) typing.remove();

    const isJarvis = sender === 'Ironcliw';
    const entry = document.createElement('div');
    entry.className = `chat-entry ${isJarvis ? 'jarvis-entry' : 'user-entry'}`;

    entry.innerHTML = `
    <div class="ce-avatar">${isJarvis ? 'J' : 'YOU'}</div>
    <div class="ce-bubble">
      <span class="ce-name">${sender}</span>
      <p>${escapeHtml(text)}</p>
    </div>
  `;
    chatLog.appendChild(entry);
    chatLog.scrollTop = chatLog.scrollHeight;
}

function showTypingIndicator() {
    const entry = document.createElement('div');
    entry.className = 'chat-entry jarvis-entry typing-entry';
    entry.innerHTML = `
    <div class="ce-avatar">J</div>
    <div class="ce-bubble">
      <span class="ce-name">Ironcliw</span>
      <div class="typing-dots">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
    chatLog.appendChild(entry);
    chatLog.scrollTop = chatLog.scrollHeight;
}

document.getElementById('clear-chat-btn').addEventListener('click', () => {
    chatLog.innerHTML = '';
    addChatMessage('IRONCLIW', 'Chat cleared. How can I help you?');
});

// ─────────────────────────────────────────────
// 9. ACTION LOG (left panel)
// ─────────────────────────────────────────────
const actionLogEl = document.getElementById('action-log');

function addAction(text) {
    // Remove placeholder
    const placeholder = actionLogEl.querySelector('.no-data-msg');
    if (placeholder) placeholder.remove();

    const now = new Date();
    const time = now.toTimeString().split(' ')[0];

    const entry = document.createElement('div');
    entry.className = 'action-entry';
    entry.innerHTML = `<div class="ae-time">${time}</div><div>${escapeHtml(text)}</div>`;
    actionLogEl.prepend(entry);

    // Keep only last 20
    const entries = actionLogEl.querySelectorAll('.action-entry');
    if (entries.length > 20) entries[entries.length - 1].remove();
}

// ─────────────────────────────────────────────
// 10. MIC BUTTON — Push to Talk / Toggle
// ─────────────────────────────────────────────
let micActive = false;
let recognition = null;

const micBtn = document.getElementById('mic-btn');

micBtn.addEventListener('click', () => {
    if (!micActive) {
        startMic();
    } else {
        stopMic();
    }
});

function startMic() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        showToast('Speech recognition not available in this build', 'warn');
        // Still signal the backend to start listening
        sendToBackend({ type: 'start_listening' });
        startListeningState();
        micActive = true;
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
        micActive = true;
        startListeningState();
        sendToBackend({ type: 'start_listening' });
    };

    recognition.onresult = (event) => {
        const last = event.results[event.results.length - 1];
        const transcript = last[0].transcript.trim();
        if (last.isFinal && transcript) {
            handleUserInput(transcript);
        }
    };

    recognition.onend = () => {
        micActive = false;
        stopListeningState();
    };

    recognition.onerror = (e) => {
        micActive = false;
        stopListeningState();
        if (e.error !== 'no-speech') showToast('Mic error: ' + e.error, 'error');
    };

    recognition.start();
}

function stopMic() {
    micActive = false;
    if (recognition) {
        try { recognition.stop(); } catch (e) { }
        recognition = null;
    }
    stopListeningState();
    sendToBackend({ type: 'stop_listening' });
}

// ─────────────────────────────────────────────
// 11. TEXT INPUT
// ─────────────────────────────────────────────
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');

sendBtn.addEventListener('click', () => submitText());
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitText();
    }
});

function submitText() {
    const text = textInput.value.trim();
    if (!text) return;
    textInput.value = '';
    handleUserInput(text);
}

// ─────────────────────────────────────────────
// 12. USER INPUT HANDLER
// ─────────────────────────────────────────────
function handleUserInput(text) {
    addChatMessage('YOU', text);
    setJarvisState('THINKING', 'thinking');
    showTypingIndicator();
    addAction(`User: "${text.substring(0, 40)}${text.length > 40 ? '…' : ''}"`);

    if (!sendToBackend({ type: 'user_message', message: text, text })) {
        // Fallback local response when backend offline
        setTimeout(() => {
            const reply = getLocalResponse(text);
            addChatMessage('IRONCLIW', reply);
            setIroncliwState('STANDBY', 'active');
        }, 900);
    }
}

function getLocalResponse(input) {
    const t = input.toLowerCase();
    if (t.includes('hello') || t.includes('hi')) return 'Hello! I\'m online but backend is currently offline. Some features may be limited.';
    if (t.includes('time')) return `Current time: ${new Date().toLocaleTimeString()}`;
    if (t.includes('date')) return `Today is ${new Date().toLocaleDateString('en-IN', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}`;
    if (t.includes('status') || t.includes('how')) return 'All core systems are operational. Backend connection is offline.';
    if (t.includes('version')) return 'Ironcliw v11.0 Neural Mesh · Electron Desktop Interface';
    return 'I\'m processing that, but the backend is offline. Please start the backend with: python start_system.py';
}

// ─────────────────────────────────────────────
// 13. QUICK COMMANDS
// ─────────────────────────────────────────────
document.getElementById('quick-cmds').addEventListener('click', (e) => {
    const btn = e.target.closest('.qcmd');
    if (btn) {
        handleUserInput(btn.dataset.cmd);
        btn.style.transform = 'scale(0.92)';
        setTimeout(() => { btn.style.transform = ''; }, 150);
    }
});

// ─────────────────────────────────────────────
// 14. TOAST NOTIFICATIONS
// ─────────────────────────────────────────────
const toastContainer = document.getElementById('toast-container');

function showToast(message, type = 'info', duration = 3500) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(20px)';
        toast.style.transition = 'all 0.4s ease';
        setTimeout(() => toast.remove(), 400);
    }, duration);
}

// ─────────────────────────────────────────────
// 15. SYSTEM METRICS SIMULATION
//     (replace with real data if backend provides it)
// ─────────────────────────────────────────────
function startMetricsSim() {
    function updateMetric(id, fillId, valId, val) {
        const fill = document.getElementById(fillId);
        const valEl = document.getElementById(valId);
        if (fill) fill.style.width = val + '%';
        if (valEl) valEl.textContent = val + '%';
    }

    function tick() {
        const cpu = 15 + Math.random() * 45;
        const mem = 30 + Math.random() * 35;
        const net = 5 + Math.random() * 60;
        updateMetric(null, 'cpu-fill', 'cpu-val', cpu.toFixed(0));
        updateMetric(null, 'mem-fill', 'mem-val', mem.toFixed(0));
        updateMetric(null, 'net-fill', 'net-val', net.toFixed(0));
    }

    tick();
    setInterval(tick, 2800);
}

// ─────────────────────────────────────────────
// 16. UPTIME CLOCK
// ─────────────────────────────────────────────
let startTime = Date.now();

function startUptimeClock() {
    setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
        const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
        const s = String(elapsed % 60).padStart(2, '0');
        document.getElementById('stat-uptime').textContent = `${h}:${m}:${s}`;
    }, 1000);
}

// ─────────────────────────────────────────────
// 17. UTILITIES
// ─────────────────────────────────────────────
function escapeHtml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// ─────────────────────────────────────────────
// KEY SHORTCUTS
// ─────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
    // Ctrl+Shift+Space → Mic toggle
    if (e.ctrlKey && e.shiftKey && e.code === 'Space') {
        e.preventDefault();
        if (!micActive) startMic(); else stopMic();
    }
    // F12 → DevTools
    if (e.key === 'F12' && electronAPI) {
        electronAPI.openDevTools();
    }
    // Escape → Stop mic / clear input
    if (e.key === 'Escape') {
        if (micActive) stopMic();
    }
});

// ─────────────────────────────────────────────
// STATUS INDICATOR ON STANDBY
// ─────────────────────────────────────────────
setTimeout(() => {
    if (bootDone && currentState === 'standby') {
        setJarvisState('STANDBY', 'active');
    }
}, 4500);
