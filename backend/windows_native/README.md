# Ironcliw Windows Native Layer (C# DLLs)

This directory contains C# native code that provides Windows API access for Ironcliw, replacing the Swift bridge functionality used on macOS.

## Architecture

```
backend/windows_native/
├── JarvisWindowsNative.sln          # Visual Studio solution
├── SystemControl/                    # Window management, volume, notifications
│   ├── SystemControl.cs
│   └── SystemControl.csproj
├── ScreenCapture/                    # Screen capture using GDI+ and Windows.Graphics
│   ├── ScreenCapture.cs
│   └── ScreenCapture.csproj
├── AudioEngine/                      # WASAPI audio recording and playback
│   ├── AudioEngine.cs
│   └── AudioEngine.csproj
└── test_csharp_bindings.py          # Python integration test
```

## Components

### 1. SystemControl
**Purpose**: Windows system management functionality

**Features**:
- Window enumeration and management (User32 API)
- Window operations: focus, minimize, maximize, restore, hide, show, close
- System volume control (WinMM API)
- Toast notifications (PowerShell integration)

**API Examples**:
```csharp
var controller = new SystemController();
var windows = controller.GetAllWindows();          // List all windows
var focused = controller.GetFocusedWindow();       // Get focused window
controller.SetVolume(50);                          // Set volume to 50%
controller.ShowNotification("Title", "Message");   // Show notification
```

### 2. ScreenCapture
**Purpose**: Screen capture using Windows Graphics APIs

**Features**:
- Full screen capture using GDI+ BitBlt
- Region-specific capture with coordinates
- Window capture by handle
- Continuous capture for video/monitoring
- Multi-monitor support with MonitorInfo enumeration
- Save to PNG file

**API Examples**:
```csharp
var engine = new ScreenCaptureEngine();
byte[] screenshot = engine.CaptureScreen();        // Full screen
byte[] region = engine.CaptureRegion(0, 0, 800, 600);  // Specific region
engine.SaveScreenToFile("screenshot.png");         // Save to file

var multi = new MultiMonitorCapture();
var monitors = multi.GetAllMonitors();             // List all monitors
byte[] monitorCapture = multi.CaptureMonitor(0);   // Capture specific monitor
```

### 3. AudioEngine
**Purpose**: WASAPI audio processing

**Features**:
- Audio device enumeration (input/output)
- WASAPI audio recording with callbacks
- Audio playback (WAV format support)
- Volume control (get/set/mute)
- Device state management

**API Examples**:
```csharp
var engine = new AudioEngine();
var inputDevices = engine.GetInputDevices();       // List microphones
var outputDevices = engine.GetOutputDevices();     // List speakers
engine.StartRecording((data) => {                  // Start recording
    Console.WriteLine($"Received {data.Length} bytes");
});
engine.StopRecording();                            // Stop recording
engine.SetVolume(0.5f);                            // Set volume (0.0-1.0)
```

## Prerequisites

### 1. .NET SDK 8.0+
Download and install from: https://dotnet.microsoft.com/download

**Quick Install (Windows)**:
```powershell
# Using winget
winget install Microsoft.DotNet.SDK.8

# Or using Chocolatey
choco install dotnet-sdk
```

**Verify Installation**:
```bash
dotnet --version
# Should output: 8.0.x or higher
```

### 2. Python 3.11+
Already required by Ironcliw.

### 3. pythonnet (Python.NET)
Python package for calling .NET assemblies from Python.

```bash
pip install pythonnet
```

## Build Instructions

### Option 1: Using PowerShell Script (Recommended)
```powershell
cd backend/windows_native
.\build.ps1
```

### Option 2: Manual Build
```bash
cd backend/windows_native

# Restore NuGet packages
dotnet restore

# Build all projects in Release mode
dotnet build -c Release

# Output DLLs will be in:
# - SystemControl/bin/Release/net8.0-windows/SystemControl.dll
# - ScreenCapture/bin/Release/net8.0-windows10.0.19041.0/ScreenCapture.dll
# - AudioEngine/bin/Release/net8.0-windows/AudioEngine.dll
```

### Option 3: Build Individual Projects
```bash
dotnet build SystemControl/SystemControl.csproj -c Release
dotnet build ScreenCapture/ScreenCapture.csproj -c Release
dotnet build AudioEngine/AudioEngine.csproj -c Release
```

## Testing

### Test from Python
```bash
# Make sure C# DLLs are built first
dotnet build -c Release

# Install pythonnet
pip install pythonnet

# Run Python integration test
python test_csharp_bindings.py
```

**Expected Output**:
```
============================================================
C# Windows Native Layer - Python Bindings Test
============================================================
✅ pythonnet is installed

=== Testing SystemControl ===
📊 Getting all windows...
   Found 15 windows
   First window: Visual Studio Code (PID: 12345)
🔍 Getting focused window...
   Focused: Command Prompt
🔊 Testing volume control...
   Current volume: 50%
✅ SystemControl tests passed!

=== Testing ScreenCapture ===
📐 Getting screen size...
   Screen size: 1920x1080
📸 Capturing screen...
   Captured 2073600 bytes
💾 Saving to test_screenshot.png...
   ✅ Screenshot saved (2073600 bytes)
🖥️ Testing multi-monitor support...
   Found 2 monitor(s)
   Monitor 0: 1920x1080 at (0, 0) (PRIMARY)
   Monitor 1: 1920x1080 at (1920, 0)
✅ ScreenCapture tests passed!

=== Testing AudioEngine ===
🎤 Getting input devices...
   Found 2 input device(s)
   - Microphone (Realtek) (DEFAULT)
   - Stereo Mix
🔊 Getting output devices...
   Found 3 output device(s)
   - Speakers (Realtek) (DEFAULT)
   - HDMI Audio
   - Bluetooth Headset
🔊 Testing volume control...
   Current volume: 50%
   Muted: False
✅ AudioEngine tests passed!

============================================================
Test Summary
============================================================
SystemControl        ✅ PASSED
ScreenCapture        ✅ PASSED
AudioEngine          ✅ PASSED

🎉 All tests passed!
```

### Test Individual Components (C# Unit Tests)
```bash
# Add xUnit test projects (optional)
dotnet test
```

## Python Integration

### Calling C# from Python (Example)

```python
import clr
from pathlib import Path

# Add reference to compiled DLL
dll_path = Path("SystemControl/bin/Release/net8.0-windows/SystemControl.dll")
clr.AddReference(str(dll_path))

# Import C# namespace
from JarvisWindowsNative.SystemControl import SystemController

# Use C# class
controller = SystemController()
windows = controller.GetAllWindows()

for window in windows:
    print(f"{window.Title} (PID: {window.ProcessId}, Process: {window.ProcessName})")
```

## API Reference

### SystemControl.SystemController

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `GetAllWindows()` | - | `List<WindowInfo>` | Get all visible windows |
| `GetFocusedWindow()` | - | `WindowInfo?` | Get currently focused window |
| `FocusWindow(handle)` | `IntPtr` | `bool` | Focus window by handle |
| `MinimizeWindow(handle)` | `IntPtr` | `bool` | Minimize window |
| `MaximizeWindow(handle)` | `IntPtr` | `bool` | Maximize window |
| `RestoreWindow(handle)` | `IntPtr` | `bool` | Restore window |
| `HideWindow(handle)` | `IntPtr` | `bool` | Hide window |
| `ShowWindowHandle(handle)` | `IntPtr` | `bool` | Show window |
| `CloseWindowHandle(handle)` | `IntPtr` | `bool` | Close window |
| `GetVolume()` | - | `int` | Get system volume (0-100) |
| `SetVolume(volume)` | `int` | `bool` | Set system volume (0-100) |
| `IncreaseVolume(amount)` | `int` | `bool` | Increase volume |
| `DecreaseVolume(amount)` | `int` | `bool` | Decrease volume |
| `ShowNotification(title, message, duration)` | `string, string, int` | `Task<bool>` | Show toast notification |

### ScreenCapture.ScreenCaptureEngine

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `CaptureScreen()` | - | `byte[]` | Capture full screen as PNG |
| `CaptureRegion(x, y, width, height)` | `int, int, int, int` | `byte[]` | Capture region as PNG |
| `CaptureWindow(handle)` | `IntPtr` | `byte[]` | Capture window as PNG |
| `SaveScreenToFile(path)` | `string` | `bool` | Save screen to file |
| `SaveWindowToFile(handle, path)` | `IntPtr, string` | `bool` | Save window to file |
| `GetScreenSize()` | - | `(int, int)` | Get screen dimensions |
| `CaptureScreenContinuous(callback, interval, token)` | `Action<byte[]>, int, CancellationToken` | `Task` | Continuous capture |

### AudioEngine.AudioEngine

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `GetInputDevices()` | - | `List<AudioDeviceInfo>` | List input devices |
| `GetOutputDevices()` | - | `List<AudioDeviceInfo>` | List output devices |
| `GetDefaultInputDevice()` | - | `AudioDeviceInfo?` | Get default input |
| `GetDefaultOutputDevice()` | - | `AudioDeviceInfo?` | Get default output |
| `StartRecording(callback, sampleRate, bitDepth)` | `Action<byte[]>, int, int` | `bool` | Start recording |
| `StopRecording()` | - | `void` | Stop recording |
| `IsRecording` | - | `bool` | Check if recording |
| `PlayAudio(data, sampleRate, channels, bitDepth)` | `byte[], int, int, int` | `bool` | Play audio |
| `StopPlayback()` | - | `void` | Stop playback |
| `IsPlaying` | - | `bool` | Check if playing |
| `GetVolume()` | - | `float` | Get volume (0.0-1.0) |
| `SetVolume(volume)` | `float` | `bool` | Set volume (0.0-1.0) |
| `IsMuted()` | - | `bool` | Check if muted |
| `SetMute(mute)` | `bool` | `bool` | Mute/unmute |
| `Dispose()` | - | `void` | Cleanup resources |

## Troubleshooting

### Build Errors

**Error**: `It was not possible to find any installed .NET SDKs`
- **Solution**: Install .NET SDK 8.0+ from https://dotnet.microsoft.com/download

**Error**: `The type or namespace name 'NAudio' could not be found`
- **Solution**: Restore NuGet packages: `dotnet restore`

**Error**: `error CS0006: Metadata file 'xxx.dll' could not be found`
- **Solution**: Clean and rebuild: `dotnet clean && dotnet build -c Release`

### Python Import Errors

**Error**: `No module named 'clr'`
- **Solution**: Install pythonnet: `pip install pythonnet`

**Error**: `Could not load file or assembly 'xxx.dll'`
- **Solution**: Make sure C# projects are built first: `dotnet build -c Release`
- Check DLL path in Python code matches actual build output location

**Error**: `System.IO.FileNotFoundException: The specified module could not be found`
- **Solution**: Missing dependencies. Make sure all NuGet packages are restored: `dotnet restore`

### Runtime Errors

**Error**: `System.UnauthorizedAccessException` (screen capture)
- **Solution**: Run with administrator privileges for some operations

**Error**: `NAudio.MmException: NoDriver` (audio)
- **Solution**: Make sure audio devices are enabled in Windows settings

## Performance Considerations

### Screen Capture
- **GDI+ BitBlt**: ~10-15ms for 1920x1080 screen (60+ FPS possible)
- **Multi-monitor**: Linear scaling per monitor
- **Optimization**: Use `CaptureRegion()` instead of full screen when possible

### Audio
- **Recording latency**: ~20-50ms (WASAPI shared mode)
- **Buffer size**: Default 100ms, adjustable
- **Sample rate**: 16kHz recommended for voice, 48kHz for music

### Volume Control
- **WinMM**: Legacy API, ~1-5ms latency
- **WASAPI**: Modern API, ~0.5-2ms latency (used in AudioEngine)

## Next Steps

After building and testing the C# layer:

1. ✅ Phase 2 complete (this phase)
2. ➡️ **Phase 3**: Create Python wrapper classes in `backend/platform/windows/`
3. **Phase 4**: Port Rust extensions
4. **Phase 5**: Port unified_supervisor.py

See `.zenflow/tasks/iron-cliw-0081/plan.md` for full roadmap.

## License

Same as Ironcliw main project.
