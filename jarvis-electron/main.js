const { app, BrowserWindow, ipcMain, screen, nativeTheme } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const net = require('net');

let mainWindow;
let backendProcess = null;
let backendReady = false;

// ── Backend Log File ───────────────────────────────────────────────────────────
const Ironcliw_ROOT = path.join(__dirname, '..');
const LOG_FILE = path.join(Ironcliw_ROOT, 'logs', 'electron_backend.log');

function ensureLogDir() {
    const logsDir = path.join(Ironcliw_ROOT, 'logs');
    if (!fs.existsSync(logsDir)) {
        fs.mkdirSync(logsDir, { recursive: true });
    }
}

// ── Detect Python executable ───────────────────────────────────────────────────
function getPythonPath() {
    // Windows: try 'python' first, then 'python3'
    const candidates = ['python', 'python3', 'py'];
    for (const cmd of candidates) {
        try {
            const result = require('child_process').spawnSync(cmd, ['--version'], { encoding: 'utf8' });
            if (result.status === 0) return cmd;
        } catch (e) { /* skip */ }
    }
    return 'python'; // fallback
}

// ── Start Backend (smart: skip if already running) ───────────────────────────
function startBackend() {
    // First check: is backend WebSocket already running?
    // (This happens when user starts backend first, which launches Electron automatically)
    const testSock = new net.Socket();
    testSock.setTimeout(1000);
    let alreadyRunning = false;

    testSock.on('connect', () => {
        testSock.destroy();
        alreadyRunning = true;
        console.log('[Ironcliw] Backend already running on port 8010 — connecting to existing backend');
        backendReady = true;
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('backend-status', { running: true, port: 8010, existing: true });
        }
        showToastWhenReady('Backend already running — connected!');
    });

    testSock.on('error', () => {
        testSock.destroy();
        if (!alreadyRunning) {
            // Backend NOT running — spawn it
            spawnBackend();
        }
    });

    testSock.on('timeout', () => {
        testSock.destroy();
        if (!alreadyRunning) {
            spawnBackend();
        }
    });

    testSock.connect(8010, '127.0.0.1');
}

// ── Helper: send toast to renderer when ready ─────────────────────────────────
function showToastWhenReady(msg) {
    if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('backend-log', msg);
    } else {
        // Window not ready yet — wait a bit
        setTimeout(() => {
            if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send('backend-log', msg);
            }
        }, 3000);
    }
}

// ── Spawn Python Backend Process ──────────────────────────────────────────────
function spawnBackend() {
    ensureLogDir();
    const python = getPythonPath();
    const script = path.join(Ironcliw_ROOT, 'backend', 'main.py');

    // Fallback to loading_server.py if main.py not found
    const wsScript = path.join(Ironcliw_ROOT, 'backend', 'loading_server.py');
    const entryScript = fs.existsSync(script) ? script : wsScript;

    console.log(`[Ironcliw] Starting backend: ${python} ${entryScript}`);

    const logStream = fs.createWriteStream(LOG_FILE, { flags: 'a' });
    logStream.write(`\n\n===== Ironcliw Backend started at ${new Date().toISOString()} =====\n`);

    backendProcess = spawn(python, [entryScript], {
        cwd: Ironcliw_ROOT,
        env: {
            ...process.env,
            NO_BROWSER: '1',           // Suppress any browser auto-open
            ELECTRON_APP: '1',         // Signal to backend it's running inside Electron
            Ironcliw_ELECTRON: '1',      // Extra flag for loading_server
            PYTHONUNBUFFERED: '1',     // Flush stdout immediately
        },
        detached: false,
        shell: false,
    });

    backendProcess.stdout.on('data', (data) => {
        const msg = data.toString();
        logStream.write(msg);
        process.stdout.write(`[BACKEND] ${msg}`);
        // Tell renderer about backend output
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('backend-log', msg.trim());
        }
    });

    backendProcess.stderr.on('data', (data) => {
        const msg = data.toString();
        logStream.write(`[ERR] ${msg}`);
        process.stderr.write(`[BACKEND ERR] ${msg}`);
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('backend-log', `[ERR] ${msg.trim()}`);
        }
    });

    backendProcess.on('error', (err) => {
        console.error('[Ironcliw] Failed to start backend:', err.message);
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('backend-status', { running: false, error: err.message });
        }
    });

    backendProcess.on('exit', (code, signal) => {
        console.log(`[Ironcliw] Backend exited: code=${code} signal=${signal}`);
        backendReady = false;
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('backend-status', { running: false, code, signal });
        }
    });

    // Poll HTTP port 8010 for backend readiness
    pollBackendReady(8010, 60, () => {
        console.log('[Ironcliw] Backend is ready on port 8010');
        backendReady = true;
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('backend-status', { running: true, port: 8010 });
        }
    });
}

// ── Poll until port is open ────────────────────────────────────────────────────
function pollBackendReady(port, maxAttempts, onReady) {
    let attempts = 0;
    const interval = setInterval(() => {
        attempts++;
        const sock = new net.Socket();
        sock.setTimeout(500);
        sock.on('connect', () => {
            sock.destroy();
            clearInterval(interval);
            onReady();
        });
        sock.on('error', () => sock.destroy());
        sock.on('timeout', () => sock.destroy());
        sock.connect(port, '127.0.0.1');

        if (attempts >= maxAttempts) {
            clearInterval(interval);
            console.warn(`[Ironcliw] Backend did not become ready after ${maxAttempts} attempts`);
        }
    }, 2000); // check every 2 seconds
}

// ── Stop Backend ──────────────────────────────────────────────────────────────
function stopBackend() {
    if (backendProcess && !backendProcess.killed) {
        console.log('[Ironcliw] Stopping backend process...');
        backendProcess.kill('SIGTERM');
        // Force kill after 5 seconds if still running
        setTimeout(() => {
            if (backendProcess && !backendProcess.killed) {
                backendProcess.kill('SIGKILL');
            }
        }, 5000);
    }
}

// ── Create Main Window ────────────────────────────────────────────────────────
function createWindow() {
    const { width, height } = screen.getPrimaryDisplay().workAreaSize;

    mainWindow = new BrowserWindow({
        width: 1280,
        height: 820,
        minWidth: 900,
        minHeight: 600,
        x: Math.floor((width - 1280) / 2),
        y: Math.floor((height - 820) / 2),
        frame: false,           // Frameless — custom title bar
        transparent: true,      // Allows rounded corners & glassmorphism
        backgroundColor: '#00000000',
        hasShadow: true,
        resizable: true,
        titleBarStyle: 'hidden',
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
            sandbox: false,
        },
        icon: path.join(__dirname, 'renderer', 'assets', 'icon.png'),
    });

    mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

    // Uncomment to debug:
    // mainWindow.webContents.openDevTools({ mode: 'detach' });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });

    // Maximize/restore state
    mainWindow.on('maximize', () => {
        mainWindow.webContents.send('window-maximized', true);
    });
    mainWindow.on('unmaximize', () => {
        mainWindow.webContents.send('window-maximized', false);
    });
}

// ── IPC: Window Controls ──────────────────────────────────────────────────────
ipcMain.on('window-minimize', () => {
    if (mainWindow) mainWindow.minimize();
});

ipcMain.on('window-maximize', () => {
    if (!mainWindow) return;
    if (mainWindow.isMaximized()) {
        mainWindow.unmaximize();
    } else {
        mainWindow.maximize();
    }
});

ipcMain.on('window-close', () => {
    if (mainWindow) mainWindow.close();
});

ipcMain.on('open-devtools', () => {
    if (mainWindow) mainWindow.webContents.openDevTools({ mode: 'detach' });
});

// IPC: Backend status query from renderer
ipcMain.handle('get-backend-status', () => {
    return { running: backendReady, pid: backendProcess ? backendProcess.pid : null };
});

// IPC: Restart backend from renderer
ipcMain.on('restart-backend', () => {
    console.log('[Ironcliw] Renderer requested backend restart');
    stopBackend();
    setTimeout(() => startBackend(), 2000);
});

// ── App Lifecycle ─────────────────────────────────────────────────────────────
app.whenReady().then(() => {
    // Start backend first, then open window
    startBackend();
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', () => {
    stopBackend();
    if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
    stopBackend();
});

// Prevent multiple instances
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
    app.quit();
} else {
    app.on('second-instance', () => {
        if (mainWindow) {
            if (mainWindow.isMinimized()) mainWindow.restore();
            mainWindow.focus();
        }
    });
}
