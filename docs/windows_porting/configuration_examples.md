# Ironcliw Windows Configuration Examples

## Overview

This guide provides Windows-specific configuration examples for common use cases. All configuration files use YAML format unless otherwise specified.

---

## Table of Contents

1. [Environment Variables (.env)](#environment-variables-env)
2. [Main Configuration (jarvis_config.yaml)](#main-configuration-jarvis_configyaml)
3. [Platform Configuration (windows_config.yaml)](#platform-configuration-windows_configyaml)
4. [Performance Tuning](#performance-tuning)
5. [Development vs Production](#development-vs-production)
6. [Multi-User Setup](#multi-user-setup)
7. [Cloud Integration](#cloud-integration)
8. [Custom Model Configurations](#custom-model-configurations)

---

## Environment Variables (.env)

### Minimal Configuration (Quick Start)

```bash
# .env - Minimal Windows Setup

# API Keys (Required)
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
OPENAI_API_KEY=sk-proj-your-key-here

# Platform
Ironcliw_PLATFORM=windows

# Basic Settings
Ironcliw_BACKEND_PORT=8010
Ironcliw_FRONTEND_PORT=3000
LOG_LEVEL=INFO
```

---

### Development Configuration

```bash
# .env - Development Mode

# =============================================================================
# API KEYS
# =============================================================================
ANTHROPIC_API_KEY=sk-ant-api03-your-dev-key
OPENAI_API_KEY=sk-proj-your-dev-key

# =============================================================================
# PLATFORM
# =============================================================================
Ironcliw_PLATFORM=windows
Ironcliw_IS_WINDOWS=true

# =============================================================================
# PORTS
# =============================================================================
Ironcliw_BACKEND_PORT=8010
Ironcliw_FRONTEND_PORT=3000
Ironcliw_PRIME_PORT=8000
Ironcliw_REACTOR_PORT=8090

# =============================================================================
# DEVELOPMENT MODE
# =============================================================================
Ironcliw_DEV_MODE=true
Ironcliw_HOT_RELOAD_ENABLED=true
Ironcliw_RELOAD_CHECK_INTERVAL=10
Ironcliw_RELOAD_GRACE_PERIOD=120

# Fast startup (skip heavy initialization)
FAST_START=true
Ironcliw_SKIP_HEAVY_INIT=false

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL=DEBUG
Ironcliw_VERBOSE=true
PYTHONIOENCODING=utf-8  # UTF-8 for emoji support

# =============================================================================
# AUTHENTICATION (Bypass for Development)
# =============================================================================
Ironcliw_AUTH_BYPASS=true
Ironcliw_AUTH_MODE=bypass

# =============================================================================
# FEATURES (Enable all for testing)
# =============================================================================
Ironcliw_VISION_ENABLED=true
Ironcliw_VISION_FPS=15
Ironcliw_AUDIO_ENABLED=true
Ironcliw_SYSTEM_CONTROL_ENABLED=true

# =============================================================================
# DEBUGGING
# =============================================================================
WINDOWS_NATIVE_DEBUG=true
Ironcliw_DEBUG_PLATFORM=true
Ironcliw_DEBUG_IMPORTS=true

# =============================================================================
# PATHS (Windows-specific)
# =============================================================================
Ironcliw_HOME=%USERPROFILE%\.jarvis
Ironcliw_DATA_DIR=%USERPROFILE%\.jarvis\data
Ironcliw_LOG_DIR=%USERPROFILE%\.jarvis\logs
Ironcliw_CACHE_DIR=%USERPROFILE%\.jarvis\cache

# =============================================================================
# PYTHON PATHS
# =============================================================================
PYTHONPATH=.;backend;frontend
PYTHONUNBUFFERED=1
```

---

### Production Configuration

```bash
# .env - Production Mode

# =============================================================================
# API KEYS (Use environment variables or secrets manager)
# =============================================================================
ANTHROPIC_API_KEY=sk-ant-api03-your-prod-key
OPENAI_API_KEY=sk-proj-your-prod-key

# =============================================================================
# PLATFORM
# =============================================================================
Ironcliw_PLATFORM=windows

# =============================================================================
# PORTS
# =============================================================================
Ironcliw_BACKEND_PORT=8010
Ironcliw_FRONTEND_PORT=80

# =============================================================================
# PRODUCTION MODE
# =============================================================================
Ironcliw_DEV_MODE=false
Ironcliw_HOT_RELOAD_ENABLED=false
FAST_START=false

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL=INFO
Ironcliw_VERBOSE=false

# =============================================================================
# AUTHENTICATION
# =============================================================================
Ironcliw_AUTH_BYPASS=false  # Enable when voice auth supported
Ironcliw_AUTH_MODE=windows_hello

# =============================================================================
# PERFORMANCE
# =============================================================================
Ironcliw_MEMORY_MODE=efficient
Ironcliw_MAX_WORKERS=4
Ironcliw_VISION_FPS=10

# =============================================================================
# SECURITY
# =============================================================================
Ironcliw_HTTPS_ENABLED=true
Ironcliw_REQUIRE_AUTH=true
Ironcliw_CORS_ENABLED=false  # Disable in production
Ironcliw_API_KEY_REQUIRED=true

# =============================================================================
# CLOUD OFFLOAD
# =============================================================================
Ironcliw_USE_CLOUD_ML=true
Ironcliw_LOCAL_ML_ENABLED=false  # Save RAM
Ironcliw_GCP_ENABLED=true

# =============================================================================
# MONITORING
# =============================================================================
Ironcliw_METRICS_ENABLED=true
Ironcliw_HEALTH_CHECK_INTERVAL=60
Ironcliw_CRASH_REPORTING=true
```

---

## Main Configuration (jarvis_config.yaml)

### Default Configuration

```yaml
# backend/config/jarvis_config.yaml - Windows Defaults

# =============================================================================
# GENERAL
# =============================================================================
jarvis:
  name: "Ironcliw"
  version: "1.0.0-windows"
  platform: "windows"

# =============================================================================
# API SETTINGS
# =============================================================================
api:
  host: "0.0.0.0"
  port: 8010
  cors_enabled: true
  cors_origins:
    - "http://localhost:3000"
    - "http://127.0.0.1:3000"
  max_request_size_mb: 100
  timeout_seconds: 300

# =============================================================================
# WEBSOCKET
# =============================================================================
websocket:
  enabled: true
  ping_interval: 30
  ping_timeout: 10
  max_connections: 100

# =============================================================================
# LOGGING
# =============================================================================
logging:
  level: "INFO"
  format: "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
  use_emoji: false  # Disable on Windows Console
  file:
    enabled: true
    path: "%USERPROFILE%\\.jarvis\\logs\\backend.log"
    max_size_mb: 100
    backup_count: 5
  console:
    enabled: true
    color: true

# =============================================================================
# PLATFORM
# =============================================================================
platform:
  auto_detect: true
  fallback: "windows"
  windows:
    use_native_dlls: true
    dll_path: "backend/windows_native/bin/Release"
    
# =============================================================================
# SYSTEM CONTROL
# =============================================================================
system_control:
  enabled: true
  window_management: true
  volume_control: true
  notifications: true

# =============================================================================
# AUDIO
# =============================================================================
audio:
  enabled: true
  engine: "wasapi"  # Windows-specific
  sample_rate: 16000
  channels: 1
  chunk_size: 1024
  input_device: "default"
  output_device: "default"

# =============================================================================
# VISION
# =============================================================================
vision:
  enabled: true
  engine: "gdi"  # Windows: gdi, wgc (future)
  fps: 15
  resolution: "auto"
  multi_monitor: true
  continuous_capture: false

# =============================================================================
# AUTHENTICATION
# =============================================================================
authentication:
  enabled: true
  mode: "bypass"  # Options: bypass, windows_hello, password
  voice_auth:
    enabled: false  # Not available on Windows MVP
    model: "ecapa-tdnn"
  windows_hello:
    enabled: false  # Future feature
    fallback: "password"

# =============================================================================
# PERFORMANCE
# =============================================================================
performance:
  max_workers: 4
  thread_pool_size: 8
  memory_mode: "normal"  # Options: minimal, efficient, normal
  cpu_affinity: []  # Empty = no affinity

# =============================================================================
# MEMORY
# =============================================================================
memory:
  max_cache_size_mb: 2048
  vision_buffer_frames: 60
  audio_buffer_seconds: 10
  garbage_collection_interval: 300

# =============================================================================
# HOT RELOAD (Development)
# =============================================================================
hot_reload:
  enabled: true
  method: "hash"  # Options: hash, watchdog
  check_interval_seconds: 10
  debounce_delay_seconds: 2
  grace_period_seconds: 120

# =============================================================================
# GCP INTEGRATION
# =============================================================================
gcp:
  enabled: false
  project_id: ""
  region: "us-central1"
  golden_image:
    enabled: false
    family: "jarvis-prime-golden"
  vm:
    instance_name: "jarvis-prime-node"
    machine_type: "e2-highmem-8"
    static_ip_name: "jarvis-prime-ip"

# =============================================================================
# ML INFERENCE
# =============================================================================
ml:
  local_inference: false
  cloud_inference: true
  providers:
    - "anthropic"
    - "openai"
  fallback_chain:
    - "anthropic"
    - "openai"

# =============================================================================
# FRONTEND
# =============================================================================
frontend:
  dev_server_port: 3000
  auto_rebuild: true
  hmr_enabled: true

# =============================================================================
# HEALTH CHECKS
# =============================================================================
health:
  enabled: true
  interval_seconds: 60
  endpoints:
    - "http://localhost:8010/health"
    - "http://localhost:3000"
```

---

## Platform Configuration (windows_config.yaml)

```yaml
# backend/config/windows_config.yaml - Windows-Specific

# =============================================================================
# WINDOWS NATIVE LAYER
# =============================================================================
windows_native:
  enabled: true
  dll_directory: "backend/windows_native/bin/Release"
  
  system_control:
    dll_name: "SystemControl.dll"
    class_name: "SystemControl.SystemController"
  
  screen_capture:
    dll_name: "ScreenCapture.dll"
    class_name: "ScreenCapture.ScreenCapturer"
  
  audio_engine:
    dll_name: "AudioEngine.dll"
    class_name: "AudioEngine.AudioEngine"

# =============================================================================
# WINDOWS APIS
# =============================================================================
windows_apis:
  user32:
    enabled: true
  gdi32:
    enabled: true
  winmm:
    enabled: true
  wasapi:
    enabled: true
    buffer_duration_ms: 10

# =============================================================================
# UAC (User Account Control)
# =============================================================================
uac:
  request_elevation: false  # Auto-elevate if needed
  prompt_on_startup: false
  require_admin: false

# =============================================================================
# TASK SCHEDULER
# =============================================================================
task_scheduler:
  enabled: true
  task_name: "Ironcliw\\Supervisor"
  run_on_boot: true
  restart_on_failure: true
  restart_delay_minutes: 1

# =============================================================================
# FILE SYSTEM
# =============================================================================
file_system:
  watcher_method: "hash"  # Options: hash, watchdog
  watch_extensions:
    - ".py"
    - ".yaml"
    - ".yml"
    - ".json"
    - ".js"
    - ".jsx"
    - ".ts"
    - ".tsx"
  exclude_patterns:
    - "**/__pycache__/**"
    - "**/.venv/**"
    - "**/node_modules/**"
    - "**/.git/**"

# =============================================================================
# WINDOWS DEFENDER
# =============================================================================
windows_defender:
  exclusion_recommendations:
    paths:
      - "%USERPROFILE%\\.jarvis"
      - "C:\\path\\to\\Ironcliw"
    processes:
      - "python.exe"
      - "pythonw.exe"

# =============================================================================
# PRIVACY SETTINGS
# =============================================================================
privacy:
  microphone:
    request_permission: true
    required: true
  camera:
    request_permission: true
    required: false
  location:
    request_permission: false
    required: false

# =============================================================================
# POWER MANAGEMENT
# =============================================================================
power:
  prevent_sleep_during_operation: true
  power_throttling: false  # Disable for performance

# =============================================================================
# DISPLAY
# =============================================================================
display:
  multi_monitor:
    enabled: true
    detect_on_startup: true
  dpi_awareness: "per_monitor_v2"
  scale_factor: "auto"

# =============================================================================
# PATHS (Windows-specific)
# =============================================================================
paths:
  home: "%USERPROFILE%\\.jarvis"
  data: "%USERPROFILE%\\.jarvis\\data"
  logs: "%USERPROFILE%\\.jarvis\\logs"
  cache: "%USERPROFILE%\\.jarvis\\cache"
  config: "%USERPROFILE%\\.jarvis\\config"
  models: "%USERPROFILE%\\.jarvis\\models"
  temp: "%TEMP%\\jarvis"
```

---

## Performance Tuning

### High-Performance Configuration (32GB RAM, 8+ cores)

```yaml
# jarvis_config.yaml - High-Performance

performance:
  max_workers: 8
  thread_pool_size: 16
  memory_mode: "normal"

memory:
  max_cache_size_mb: 4096
  vision_buffer_frames: 120
  audio_buffer_seconds: 30

vision:
  fps: 30  # Higher FPS
  engine: "wgc"  # Windows.Graphics.Capture (future)

audio:
  chunk_size: 512  # Smaller chunks = lower latency

ml:
  local_inference: true  # Enable if GPU available
  model_cache_size_gb: 8
```

---

### Low-Resource Configuration (16GB RAM, 4 cores)

```yaml
# jarvis_config.yaml - Low-Resource

performance:
  max_workers: 2
  thread_pool_size: 4
  memory_mode: "minimal"

memory:
  max_cache_size_mb: 512
  vision_buffer_frames: 30
  audio_buffer_seconds: 5

vision:
  fps: 10
  continuous_capture: false

audio:
  chunk_size: 2048  # Larger chunks = less overhead

ml:
  local_inference: false
  cloud_inference: true
```

---

## Development vs Production

### Development Override

```yaml
# jarvis_config_dev.yaml - Development Overrides

# Load this with: Ironcliw_CONFIG=jarvis_config_dev.yaml

jarvis:
  name: "Ironcliw-DEV"

logging:
  level: "DEBUG"
  use_emoji: true  # Windows Terminal supports this

hot_reload:
  enabled: true
  check_interval_seconds: 5  # Faster detection

authentication:
  mode: "bypass"

api:
  cors_enabled: true
  cors_origins:
    - "*"  # Allow all in dev

frontend:
  auto_rebuild: true
  hmr_enabled: true
```

---

### Production Override

```yaml
# jarvis_config_prod.yaml - Production Overrides

jarvis:
  name: "Ironcliw-PROD"

logging:
  level: "WARNING"
  file:
    max_size_mb: 500
    backup_count: 10

hot_reload:
  enabled: false

authentication:
  mode: "windows_hello"
  require_auth: true

api:
  cors_enabled: false
  max_request_size_mb: 50

memory:
  garbage_collection_interval: 600  # More frequent GC

health:
  interval_seconds: 30  # More frequent checks
```

---

## Multi-User Setup

### Shared System Configuration

```bash
# .env - Shared System (Multiple Users)

# System-wide installation
Ironcliw_INSTALL_DIR=C:\Program Files\Ironcliw
Ironcliw_SHARED_DATA=C:\ProgramData\Ironcliw

# User-specific data
Ironcliw_HOME=%USERPROFILE%\.jarvis
Ironcliw_USER_CONFIG=%USERPROFILE%\.jarvis\config

# Ports (avoid conflicts)
Ironcliw_BACKEND_PORT=8010
Ironcliw_FRONTEND_PORT=3000
```

```yaml
# jarvis_config.yaml - Multi-User

paths:
  # Shared (read-only)
  shared_models: "C:\\ProgramData\\Ironcliw\\models"
  shared_cache: "C:\\ProgramData\\Ironcliw\\cache"
  
  # User-specific (read-write)
  user_data: "%USERPROFILE%\\.jarvis\\data"
  user_logs: "%USERPROFILE%\\.jarvis\\logs"
  user_config: "%USERPROFILE%\\.jarvis\\config"

authentication:
  mode: "windows_hello"  # Per-user authentication
  require_auth: true
```

---

## Cloud Integration

### GCP Configuration

```yaml
# jarvis_config.yaml - GCP Integration

gcp:
  enabled: true
  project_id: "your-gcp-project-id"
  region: "us-central1"
  
  credentials:
    path: "%USERPROFILE%\\.jarvis\\gcp-credentials.json"
  
  golden_image:
    enabled: true
    family: "jarvis-prime-golden"
    max_age_days: 30
  
  vm:
    instance_name: "jarvis-prime-node"
    machine_type: "e2-highmem-8"
    static_ip_name: "jarvis-prime-ip"
    zone: "us-central1-a"
    
    startup_timeout: 300
    health_poll_interval: 5
  
  inference:
    use_cloud_first: true
    fallback_to_local: false
    fallback_to_claude: true
```

---

### Azure Configuration (Future)

```yaml
# jarvis_config.yaml - Azure Integration (Planned)

azure:
  enabled: false
  subscription_id: "your-subscription-id"
  resource_group: "jarvis-rg"
  
  vm:
    name: "jarvis-vm"
    size: "Standard_D8s_v3"
    region: "eastus"
  
  cognitive_services:
    speech:
      enabled: true
      key: "${AZURE_SPEECH_KEY}"
      region: "eastus"
```

---

## Custom Model Configurations

### Local Model Configuration

```yaml
# jarvis_config.yaml - Local ML Models

ml:
  local_inference: true
  
  models:
    llm:
      provider: "llamacpp"
      model_path: "%USERPROFILE%\\.jarvis\\models\\mistral-7b-q4.gguf"
      context_size: 4096
      gpu_layers: 0  # CPU-only on Windows MVP
    
    vision:
      provider: "yolo"
      model_path: "%USERPROFILE%\\.jarvis\\models\\yolov8n.pt"
      confidence_threshold: 0.5
    
    audio:
      provider: "whisper"
      model_path: "%USERPROFILE%\\.jarvis\\models\\whisper-base.pt"
      language: "en"
```

---

### Cloud Model Configuration

```yaml
# jarvis_config.yaml - Cloud ML Models

ml:
  local_inference: false
  cloud_inference: true
  
  providers:
    anthropic:
      enabled: true
      api_key: "${ANTHROPIC_API_KEY}"
      model: "claude-3-5-sonnet-20241022"
      max_tokens: 4096
      temperature: 0.7
    
    openai:
      enabled: true
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4-turbo-preview"
      max_tokens: 4096
      temperature: 0.7
  
  fallback_chain:
    - "anthropic"
    - "openai"
  
  retry:
    max_attempts: 3
    backoff_factor: 2
```

---

## Loading Configurations

### Environment-Based Loading

```powershell
# Load development config
$env:Ironcliw_CONFIG="jarvis_config_dev.yaml"
python unified_supervisor.py

# Load production config
$env:Ironcliw_CONFIG="jarvis_config_prod.yaml"
python unified_supervisor.py

# Load custom config
$env:Ironcliw_CONFIG="C:\path\to\custom_config.yaml"
python unified_supervisor.py
```

---

### Merge Multiple Configs

```python
# custom_startup.py - Merge multiple configs

import yaml

# Load base config
with open('backend/config/jarvis_config.yaml') as f:
    base_config = yaml.safe_load(f)

# Load Windows-specific config
with open('backend/config/windows_config.yaml') as f:
    windows_config = yaml.safe_load(f)

# Load user overrides
with open(os.path.expandvars('%USERPROFILE%\\.jarvis\\config\\user_config.yaml')) as f:
    user_config = yaml.safe_load(f)

# Merge (user > windows > base)
from backend.utils.config_merger import deep_merge
final_config = deep_merge(base_config, windows_config, user_config)
```

---

## Validation

### Validate Configuration

```powershell
# Validate config syntax (YAML)
python -c "import yaml; yaml.safe_load(open('backend/config/jarvis_config.yaml'))"

# Validate config completeness
python unified_supervisor.py --validate-config

# Generate default config
python unified_supervisor.py --generate-default-config > my_config.yaml
```

---

**Last Updated:** February 2026  
**Windows Port Version:** 1.0.0-MVP
