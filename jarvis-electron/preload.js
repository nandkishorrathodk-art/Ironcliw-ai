const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    // ── Window Controls ────────────────────────────────────────────────────
    minimize: () => ipcRenderer.send('window-minimize'),
    maximize: () => ipcRenderer.send('window-maximize'),
    close: () => ipcRenderer.send('window-close'),
    openDevTools: () => ipcRenderer.send('open-devtools'),

    // ── Platform info ──────────────────────────────────────────────────────
    platform: process.platform,
    versions: {
        electron: process.versions.electron,
        node: process.versions.node,
        chrome: process.versions.chrome,
    },

    // ── Window state listeners ─────────────────────────────────────────────
    onWindowMaximized: (callback) => {
        ipcRenderer.on('window-maximized', (_event, isMaximized) => callback(isMaximized));
    },

    // ── Backend Management ─────────────────────────────────────────────────
    getBackendStatus: () => ipcRenderer.invoke('get-backend-status'),
    restartBackend: () => ipcRenderer.send('restart-backend'),

    // Backend status updates from main process
    onBackendStatus: (callback) => {
        ipcRenderer.on('backend-status', (_event, status) => callback(status));
    },

    // Backend log stream (optional: display in action log)
    onBackendLog: (callback) => {
        ipcRenderer.on('backend-log', (_event, line) => callback(line));
    },
});
