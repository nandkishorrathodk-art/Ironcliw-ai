# Ironcliw Linux Setup Guide

**Complete installation guide for Ironcliw on Ubuntu, Debian, Fedora, and Arch Linux**

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [System Requirements](#system-requirements)
- [Installation by Distribution](#installation-by-distribution)
- [Configuration](#configuration)
- [First Run](#first-run)
- [Troubleshooting](#troubleshooting)
- [Common Issues](#common-issues)

---

## Prerequisites

### Supported Distributions

✅ **Tested and supported**:
- Ubuntu 20.04+ / Linux Mint 20+
- Debian 11+ (Bullseye)
- Fedora 35+
- Arch Linux (rolling)
- Pop!_OS 21.04+
- Manjaro (rolling)

✅ **Desktop Environments**:
- GNOME 3.38+
- KDE Plasma 5.20+
- XFCE 4.16+
- i3 / Sway (tiling window managers)

✅ **Display Servers**:
- X11 (Xorg) - fully supported
- Wayland - supported with limitations

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Distribution** | Ubuntu 20.04 | Ubuntu 22.04+ |
| **RAM** | 8 GB | 16 GB+ |
| **Storage** | 10 GB free | 20 GB+ free (SSD preferred) |
| **CPU** | 4 cores | 8+ cores |
| **GPU** | None (CPU fallback) | NVIDIA/AMD GPU with driver support |
| **Network** | Internet connection | High-speed internet for cloud features |

---

## Installation by Distribution

### Ubuntu / Debian / Linux Mint

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3 python3-pip python3-venv \
    git build-essential \
    portaudio19-dev python3-pyaudio \
    espeak-ng libespeak-ng1 \
    xdotool wmctrl scrot \
    libgirepository1.0-dev gir1.2-gtk-3.0 \
    libcairo2-dev pkg-config python3-dev \
    libdbus-1-dev libglib2.0-dev

# For GPU support (NVIDIA)
# sudo apt install nvidia-driver-535 nvidia-cuda-toolkit

# For GPU support (AMD)
# sudo apt install mesa-opencl-icd rocm-opencl-runtime

# Clone Ironcliw
cd ~/Documents
git clone https://github.com/drussell23/Ironcliw-AI-Agent.git
cd Ironcliw-AI-Agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Fedora / Red Hat / CentOS

```bash
# Update system
sudo dnf update -y

# Install system dependencies
sudo dnf install -y \
    python3 python3-pip python3-virtualenv \
    git gcc gcc-c++ make \
    portaudio-devel python3-pyaudio \
    espeak-ng \
    xdotool wmctrl scrot \
    gobject-introspection-devel gtk3 \
    cairo-devel pkg-config python3-devel \
    dbus-devel glib2-devel

# For GPU support (NVIDIA)
# sudo dnf install akmod-nvidia nvidia-driver-cuda

# Clone Ironcliw
cd ~/Documents
git clone https://github.com/drussell23/Ironcliw-AI-Agent.git
cd Ironcliw-AI-Agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Arch Linux / Manjaro

```bash
# Update system
sudo pacman -Syu

# Install system dependencies
sudo pacman -S --needed \
    python python-pip python-virtualenv \
    git base-devel \
    portaudio python-pyaudio \
    espeak-ng \
    xdotool wmctrl scrot \
    gobject-introspection gtk3 \
    cairo pkg-config \
    dbus glib2

# For GPU support (NVIDIA)
# sudo pacman -S nvidia nvidia-utils cuda

# For GPU support (AMD)
# sudo pacman -S rocm-opencl-runtime

# Clone Ironcliw
cd ~/Documents
git clone https://github.com/drussell23/Ironcliw-AI-Agent.git
cd Ironcliw-AI-Agent

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

---

## Configuration

### 1. Create Environment File

```bash
# Copy platform template
cp .env.platform.example .env
```

### 2. Edit Configuration

Edit `.env` with your preferred text editor (nano, vim, gedit, etc.):

```bash
nano .env
```

Configure for Linux:

```bash
# Platform Configuration
Ironcliw_PLATFORM=linux

# Authentication (bypass enabled for Linux by default)
Ironcliw_AUTH_BYPASS=true

# TTS Engine (Linux uses espeak)
Ironcliw_TTS_ENGINE=pyttsx3

# Screen Capture Method
Ironcliw_CAPTURE_METHOD=mss

# Display Server
Ironcliw_DISPLAY_SERVER=auto  # Options: x11, wayland, auto

# GPU Backend
Ironcliw_GPU_BACKEND=auto  # Options: cpu, cuda, rocm, vulkan

# ML Inference Backend
Ironcliw_ML_BACKEND=cpu  # Change to 'cuda' for NVIDIA or 'rocm' for AMD

# Data Directories (Linux XDG-compliant paths)
Ironcliw_CONFIG_DIR=$HOME/.config/jarvis
Ironcliw_LOG_DIR=$HOME/.local/share/jarvis/logs
Ironcliw_DATA_DIR=$HOME/.local/share/jarvis/data
Ironcliw_CACHE_DIR=$HOME/.cache/jarvis
```

### 3. Configure Linux-Specific Settings

Edit `backend/config/linux_config.yaml`:

```yaml
# Text-to-Speech Configuration
tts:
  engine: espeak-ng  # Or 'festival', 'pico2wave'
  voice: en-us  # Language/accent
  rate: 175  # Speech rate (words per minute)

# Screen Capture
screen_capture:
  method: mss  # Fast screen capture
  wayland_tool: grim  # For Wayland: grim, gnome-screenshot
  x11_fallback: true

# GPU Configuration
gpu:
  backend: auto  # Auto-detect: cuda, rocm, vulkan
  device_id: 0

# Desktop Environment
desktop:
  type: auto  # Options: gnome, kde, xfce, i3, sway, auto
  compositor: auto  # Wayland compositor
```

### 4. Set Permissions

```bash
# Make data directories (created automatically on first run)
mkdir -p ~/.config/jarvis
mkdir -p ~/.local/share/jarvis/{logs,data}
mkdir -p ~/.cache/jarvis

# Set proper permissions
chmod 700 ~/.config/jarvis
chmod 755 ~/.local/share/jarvis
```

---

## First Run

### 1. Start Ironcliw

**Important**: Always run from activated virtual environment!

```bash
# Navigate to Ironcliw directory
cd ~/Documents/Ironcliw-AI-Agent

# Activate virtual environment
source venv/bin/activate

# Start Ironcliw supervisor
python3 unified_supervisor.py
```

### 2. What to Expect

First startup will:
1. ✅ Detect Linux distribution and desktop environment
2. ✅ Initialize cross-platform abstractions
3. ✅ Start backend server (port 8010)
4. ✅ Start frontend UI (port 3000)
5. ✅ Open browser to http://localhost:3000

**Startup time**: 30-60 seconds on first run

### 3. Verify Everything Works

Once the UI loads, you should see:
- ✅ "Ironcliw READY" status
- ✅ System status indicators (all green)
- ✅ No error messages in the console

### 4. Test Basic Features

Try these commands in the UI:
- "What's my screen resolution?" (tests screen capture)
- "List all windows" (tests window management)
- "Say hello" (tests text-to-speech)

---

## Troubleshooting

### Backend Won't Start

**Symptom**: Port 8010 already in use

**Solution**:
```bash
# Find process using port 8010
lsof -i :8010

# Kill the process (replace PID with actual process ID)
kill -9 <PID>

# Or kill all Python processes (if safe)
pkill -f "python.*unified_supervisor"
```

### Frontend Won't Load

**Symptom**: Port 3000 already in use

**Solution**:
```bash
# Find process using port 3000
lsof -i :3000

# Kill the process
kill -9 <PID>
```

### Python Virtual Environment Issues

**Symptom**: Cannot create or activate virtual environment

**Solution**:
```bash
# Install python3-venv if missing (Ubuntu/Debian)
sudo apt install python3-venv

# Or use virtualenv (Fedora/Arch)
pip install virtualenv
virtualenv venv

# Activate with source (not .)
source venv/bin/activate
```

### Dependencies Won't Install

**Symptom**: `pip install` fails with compilation errors

**Solution**:
```bash
# Install build dependencies (Ubuntu/Debian)
sudo apt install build-essential python3-dev

# Install build dependencies (Fedora)
sudo dnf install gcc gcc-c++ python3-devel

# Install build dependencies (Arch)
sudo pacman -S base-devel python

# Try installing with verbose output to see exact error
pip install -r requirements.txt --verbose
```

### Screen Capture Not Working

**Symptom**: Black screen or capture errors

**Solution for X11**:
```bash
# Verify X11 is running
echo $DISPLAY  # Should output :0 or similar

# Install xrandr if missing
sudo apt install x11-xserver-utils

# Test screen capture manually
python3 -c "from mss import mss; mss().shot()"
```

**Solution for Wayland**:
```bash
# Install grim for Wayland screen capture
sudo apt install grim  # Ubuntu/Debian
sudo dnf install grim  # Fedora
sudo pacman -S grim   # Arch

# Or fall back to X11
# Edit .env: Ironcliw_DISPLAY_SERVER=x11
```

### TTS Not Speaking

**Symptom**: No voice output

**Solution**:
```bash
# Test espeak directly
espeak-ng "Hello, this is a test"

# If espeak not found, install it
sudo apt install espeak-ng  # Ubuntu/Debian
sudo dnf install espeak-ng  # Fedora
sudo pacman -S espeak-ng   # Arch

# Test pyttsx3
python3 -c "import pyttsx3; engine = pyttsx3.init(); engine.say('test'); engine.runAndWait()"

# Check ALSA/PulseAudio configuration
pactl info  # Verify PulseAudio is running
alsamixer   # Check volume levels
```

### GPU Not Detected

**Symptom**: Ironcliw falls back to CPU despite having GPU

**Solution for NVIDIA**:
```bash
# Verify NVIDIA driver is installed
nvidia-smi

# Install CUDA toolkit (Ubuntu/Debian)
sudo apt install nvidia-cuda-toolkit

# Install PyTorch with CUDA support
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set GPU backend in .env
# Ironcliw_GPU_BACKEND=cuda
# Ironcliw_ML_BACKEND=cuda
```

**Solution for AMD**:
```bash
# Install ROCm (Ubuntu/Debian)
sudo apt install rocm-opencl-runtime

# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# Set GPU backend in .env
# Ironcliw_GPU_BACKEND=rocm
# Ironcliw_ML_BACKEND=rocm
```

### Window Management Not Working

**Symptom**: Cannot list or focus windows

**Solution**:
```bash
# Install wmctrl and xdotool
sudo apt install wmctrl xdotool  # Ubuntu/Debian
sudo dnf install wmctrl xdotool  # Fedora
sudo pacman -S wmctrl xdotool   # Arch

# Test manually
wmctrl -l  # List all windows
xdotool search --name "Firefox"  # Search for Firefox window
```

---

## Common Issues

### Issue: Permission Denied Errors

**Symptom**: Cannot access files or directories

**Solution**:
```bash
# Fix ownership of Ironcliw directory
sudo chown -R $USER:$USER ~/Documents/Ironcliw-AI-Agent

# Fix permissions
chmod -R 755 ~/Documents/Ironcliw-AI-Agent
```

### Issue: Import Errors After Installation

**Symptom**: `ModuleNotFoundError` despite installing dependencies

**Solution**:
```bash
# Ensure you're in virtual environment
source venv/bin/activate

# Verify which Python is being used
which python3  # Should point to venv/bin/python3

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: Slow Performance

**Symptom**: Ironcliw is slow or unresponsive

**Solutions**:
1. **Enable GPU acceleration** (if available):
   ```bash
   # Edit .env
   # Ironcliw_GPU_BACKEND=cuda  # Or rocm for AMD
   # Ironcliw_ML_BACKEND=cuda
   ```

2. **Reduce screen capture FPS**:
   ```bash
   # Edit backend/config/linux_config.yaml
   # screen_capture.fps: 15
   ```

3. **Use X11 instead of Wayland** (Wayland has overhead):
   ```bash
   # Edit .env
   # Ironcliw_DISPLAY_SERVER=x11
   
   # Or logout and select X11 session at login
   ```

4. **Close unnecessary applications**:
   ```bash
   # Check memory usage
   free -h
   
   # Check CPU usage
   htop
   ```

### Issue: Wayland Compatibility Issues

**Symptom**: Screen capture or window management broken on Wayland

**Solution**:
```bash
# Option 1: Use X11 session instead
# Logout and select "Ubuntu on Xorg" (or similar) at login screen

# Option 2: Install Wayland tools
sudo apt install grim slurp wl-clipboard  # Screen capture tools

# Option 3: Enable XWayland fallback
# Edit .env: Ironcliw_DISPLAY_SERVER=x11
```

### Issue: System Tray Icon Not Showing

**Symptom**: Ironcliw tray icon missing

**Solution**:
```bash
# Install AppIndicator support (GNOME)
sudo apt install gnome-shell-extension-appindicator

# Enable extension
gnome-extensions enable appindicatorsupport@rgcjonas.gmail.com

# Restart GNOME Shell: Alt+F2, type 'r', press Enter

# For KDE Plasma (system tray widget must be added to panel)
# Right-click panel > Add Widgets > System Tray
```

---

## Desktop Environment Specific Notes

### GNOME

- ✅ Fully supported with X11 and Wayland
- Screen capture on Wayland requires `grim` or GNOME Screenshot
- System tray requires AppIndicator extension

### KDE Plasma

- ✅ Fully supported with X11 and Wayland
- Window management uses `wmctrl` on X11
- System tray works out of the box

### XFCE

- ✅ Fully supported (X11 only)
- Lightweight and fast
- System tray works natively

### i3 / Sway

- ✅ Supported tiling window managers
- i3: X11-based, fully functional
- Sway: Wayland-based, requires `grim` for screen capture
- Use `i3-msg` for window management

---

## Next Steps

✅ **Setup complete!** Ironcliw is now running on Linux.

**Learn more**:
- [Main README](../../README.md) - Overview and features
- [API Documentation](../API.md) - REST and WebSocket APIs
- [Architecture Guide](../architecture/) - System architecture
- [Windows Setup](WINDOWS_SETUP.md) - Install on Windows

**Join the community**:
- GitHub Issues: Report bugs or request features
- Discussions: Ask questions and share experiences

---

## Advanced Configuration

### Multi-Monitor Setup

```yaml
# backend/config/linux_config.yaml
screen_capture:
  monitor: 0  # Primary monitor
  # Or specify monitor by index: 1, 2, etc.
  # Use -1 for all monitors
```

### Custom Data Directories

```bash
# Create custom directories
mkdir -p /mnt/data/jarvis/{logs,data,cache}

# Update .env
# Ironcliw_DATA_DIR=/mnt/data/jarvis/data
# Ironcliw_LOG_DIR=/mnt/data/jarvis/logs
# Ironcliw_CACHE_DIR=/mnt/data/jarvis/cache
```

### Systemd Service (Auto-start on Boot)

Create `/etc/systemd/system/jarvis.service`:

```ini
[Unit]
Description=Ironcliw AI Assistant
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/Documents/Ironcliw-AI-Agent
ExecStart=/home/youruser/Documents/Ironcliw-AI-Agent/venv/bin/python unified_supervisor.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable jarvis
sudo systemctl start jarvis
sudo systemctl status jarvis
```

---

## Uninstallation

To completely remove Ironcliw:

```bash
# 1. Deactivate virtual environment
deactivate

# 2. Delete Ironcliw directory
cd ~
rm -rf ~/Documents/Ironcliw-AI-Agent

# 3. Delete user data (optional)
rm -rf ~/.config/jarvis
rm -rf ~/.local/share/jarvis
rm -rf ~/.cache/jarvis

# 4. Remove system dependencies (optional, if not used by other apps)
sudo apt autoremove
```

---

**Last updated**: February 2026  
**Version**: 1.0.0 (Cross-Platform Release)  
**Platforms**: Ubuntu, Debian, Fedora, Arch Linux
