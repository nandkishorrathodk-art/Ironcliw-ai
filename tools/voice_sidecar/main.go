package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/mem"
	"golang.org/x/sync/singleflight"
	"gopkg.in/yaml.v3"
)

type Config struct {
	Listen           ListenConfig   `json:"listen" yaml:"listen"`
	Pressure         PressureConfig `json:"pressure" yaml:"pressure"`
	Worker           WorkerConfig   `json:"worker" yaml:"worker"`
	Backoff          BackoffConfig  `json:"backoff" yaml:"backoff"`
	ControlTimeoutMs int            `json:"control_timeout_ms" yaml:"control_timeout_ms"`
}

type ListenConfig struct {
	Network string `json:"network" yaml:"network"`
	Address string `json:"address" yaml:"address"`
}

type PressureConfig struct {
	FailClosed          bool    `json:"fail_closed" yaml:"fail_closed"`
	MinAvailableMemoryMB uint64  `json:"min_available_memory_mb" yaml:"min_available_memory_mb"`
	MaxMemoryPercent    float64 `json:"max_memory_percent" yaml:"max_memory_percent"`
	MaxCPUPercent       float64 `json:"max_cpu_percent" yaml:"max_cpu_percent"`
	MonitorIntervalMs   int     `json:"monitor_interval_ms" yaml:"monitor_interval_ms"`
	SampleTimeoutMs     int     `json:"sample_timeout_ms" yaml:"sample_timeout_ms"`
}

type WorkerConfig struct {
	Command              []string          `json:"command" yaml:"command"`
	WorkingDir           string            `json:"working_dir" yaml:"working_dir"`
	Env                  map[string]string `json:"env" yaml:"env"`
	AutoStart            bool              `json:"auto_start" yaml:"auto_start"`
	AutoRecover          bool              `json:"auto_recover" yaml:"auto_recover"`
	StopOnPressure       bool              `json:"stop_on_pressure" yaml:"stop_on_pressure"`
	HealthURL            string            `json:"health_url" yaml:"health_url"`
	HealthTimeoutMs      int               `json:"health_timeout_ms" yaml:"health_timeout_ms"`
	HealthFailureLimit   int               `json:"health_failure_limit" yaml:"health_failure_limit"`
	StartupHealthWaitMs  int               `json:"startup_health_wait_ms" yaml:"startup_health_wait_ms"`
	StartupHealthPollMs  int               `json:"startup_health_poll_ms" yaml:"startup_health_poll_ms"`
	StopTimeoutMs        int               `json:"stop_timeout_ms" yaml:"stop_timeout_ms"`
}

type BackoffConfig struct {
	InitialMs  int     `json:"initial_ms" yaml:"initial_ms"`
	MaxMs      int     `json:"max_ms" yaml:"max_ms"`
	Multiplier float64 `json:"multiplier" yaml:"multiplier"`
}

type PressureSnapshot struct {
	Timestamp          time.Time `json:"timestamp"`
	CPUPercent         float64   `json:"cpu_percent"`
	MemoryPercent      float64   `json:"memory_percent"`
	MemoryAvailableMB  uint64    `json:"memory_available_mb"`
	MemoryTotalMB      uint64    `json:"memory_total_mb"`
	Valid              bool      `json:"valid"`
	Reason             string    `json:"reason,omitempty"`
}

type WorkerStatus struct {
	State             string    `json:"state"`
	PID               int       `json:"pid"`
	StartedAt         time.Time `json:"started_at,omitempty"`
	RestartCount      int       `json:"restart_count"`
	LastExit          string    `json:"last_exit,omitempty"`
	LastError         string    `json:"last_error,omitempty"`
	BackoffUntil      time.Time `json:"backoff_until,omitempty"`
	HealthFailures    int       `json:"health_failures"`
	LastHealthCheckAt time.Time `json:"last_health_check_at,omitempty"`
}

type Metrics struct {
	StartRequests   uint64
	StopRequests    uint64
	RestartRequests uint64
	CrashRecoveries uint64
	CrashCount      uint64
	PressureSamples uint64
}

type ControlRequest struct {
	Reason string `json:"reason"`
}

type Supervisor struct {
	cfg    Config
	logger *slog.Logger

	httpServer *http.Server
	listener   net.Listener

	runCtx    context.Context
	runCancel context.CancelFunc
	stopOnce  sync.Once
	wg        sync.WaitGroup

	mu sync.RWMutex

	workerCmd           *exec.Cmd
	workerPID           int
	workerStartedAt     time.Time
	workerStopRequested bool
	workerRestartCount  int
	workerLastExit      string
	workerLastError     string
	workerBackoffUntil  time.Time
	backoffExp          int

	pressure              PressureSnapshot
	healthFailureCount    int
	lastHealthCheckAt     time.Time
	workerHealthCheckPass bool

	metrics Metrics

	sf singleflight.Group

	samplePressureFn    func(context.Context) (PressureSnapshot, error)
	checkWorkerHealthFn func(context.Context, string, time.Duration) error

	serveErr atomic.Value
}

func defaultConfig() Config {
	return Config{
		Listen: ListenConfig{
			Network: "tcp",
			Address: "127.0.0.1:9860",
		},
		Pressure: PressureConfig{
			FailClosed:           true,
			MinAvailableMemoryMB: 2048,
			MaxMemoryPercent:     85.0,
			MaxCPUPercent:        90.0,
			MonitorIntervalMs:    1000,
			SampleTimeoutMs:      800,
		},
		Worker: WorkerConfig{
			Command:             []string{},
			WorkingDir:          "",
			Env:                 map[string]string{},
			AutoStart:           false,
			AutoRecover:         true,
			StopOnPressure:      true,
			HealthURL:           "",
			HealthTimeoutMs:     1500,
			HealthFailureLimit:  3,
			StartupHealthWaitMs: 15000,
			StartupHealthPollMs: 250,
			StopTimeoutMs:       5000,
		},
		Backoff: BackoffConfig{
			InitialMs:  1000,
			MaxMs:      30000,
			Multiplier: 2.0,
		},
		ControlTimeoutMs: 5000,
	}
}

func loadConfig(path string) (Config, error) {
	cfg := defaultConfig()
	if strings.TrimSpace(path) == "" {
		applyEnvOverrides(&cfg)
		return cfg, validateConfig(cfg)
	}

	buf, err := os.ReadFile(path)
	if err != nil {
		return cfg, fmt.Errorf("read config: %w", err)
	}

	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".json":
		if err := json.Unmarshal(buf, &cfg); err != nil {
			return cfg, fmt.Errorf("parse json config: %w", err)
		}
	default:
		if err := yaml.Unmarshal(buf, &cfg); err != nil {
			return cfg, fmt.Errorf("parse yaml config: %w", err)
		}
	}

	applyEnvOverrides(&cfg)
	if err := validateConfig(cfg); err != nil {
		return cfg, err
	}
	return cfg, nil
}

func applyEnvOverrides(cfg *Config) {
	overrideString(&cfg.Listen.Network, "JARVIS_VOICE_SIDECAR_NETWORK")
	overrideString(&cfg.Listen.Address, "JARVIS_VOICE_SIDECAR_ADDRESS")

	overrideBool(&cfg.Pressure.FailClosed, "JARVIS_VOICE_SIDECAR_FAIL_CLOSED")
	overrideUint64(&cfg.Pressure.MinAvailableMemoryMB, "JARVIS_VOICE_SIDECAR_MIN_MEM_MB")
	overrideFloat64(&cfg.Pressure.MaxMemoryPercent, "JARVIS_VOICE_SIDECAR_MAX_MEM_PERCENT")
	overrideFloat64(&cfg.Pressure.MaxCPUPercent, "JARVIS_VOICE_SIDECAR_MAX_CPU_PERCENT")
	overrideInt(&cfg.Pressure.MonitorIntervalMs, "JARVIS_VOICE_SIDECAR_MONITOR_INTERVAL_MS")
	overrideInt(&cfg.Pressure.SampleTimeoutMs, "JARVIS_VOICE_SIDECAR_SAMPLE_TIMEOUT_MS")

	if raw := strings.TrimSpace(os.Getenv("JARVIS_VOICE_SIDECAR_WORKER_COMMAND")); raw != "" {
		cfg.Worker.Command = strings.Fields(raw)
	}
	overrideString(&cfg.Worker.WorkingDir, "JARVIS_VOICE_SIDECAR_WORKER_DIR")
	overrideBool(&cfg.Worker.AutoStart, "JARVIS_VOICE_SIDECAR_WORKER_AUTOSTART")
	overrideBool(&cfg.Worker.AutoRecover, "JARVIS_VOICE_SIDECAR_WORKER_AUTORECOVER")
	overrideBool(&cfg.Worker.StopOnPressure, "JARVIS_VOICE_SIDECAR_STOP_ON_PRESSURE")
	overrideString(&cfg.Worker.HealthURL, "JARVIS_VOICE_SIDECAR_WORKER_HEALTH_URL")
	overrideInt(&cfg.Worker.HealthTimeoutMs, "JARVIS_VOICE_SIDECAR_WORKER_HEALTH_TIMEOUT_MS")
	overrideInt(&cfg.Worker.HealthFailureLimit, "JARVIS_VOICE_SIDECAR_WORKER_HEALTH_FAILURE_LIMIT")
	overrideInt(&cfg.Worker.StartupHealthWaitMs, "JARVIS_VOICE_SIDECAR_WORKER_STARTUP_WAIT_MS")
	overrideInt(&cfg.Worker.StartupHealthPollMs, "JARVIS_VOICE_SIDECAR_WORKER_STARTUP_POLL_MS")
	overrideInt(&cfg.Worker.StopTimeoutMs, "JARVIS_VOICE_SIDECAR_WORKER_STOP_TIMEOUT_MS")

	overrideInt(&cfg.Backoff.InitialMs, "JARVIS_VOICE_SIDECAR_BACKOFF_INITIAL_MS")
	overrideInt(&cfg.Backoff.MaxMs, "JARVIS_VOICE_SIDECAR_BACKOFF_MAX_MS")
	overrideFloat64(&cfg.Backoff.Multiplier, "JARVIS_VOICE_SIDECAR_BACKOFF_MULTIPLIER")
	overrideInt(&cfg.ControlTimeoutMs, "JARVIS_VOICE_SIDECAR_CONTROL_TIMEOUT_MS")
}

func overrideString(target *string, env string) {
	if v := strings.TrimSpace(os.Getenv(env)); v != "" {
		*target = v
	}
}

func overrideBool(target *bool, env string) {
	if v := strings.TrimSpace(strings.ToLower(os.Getenv(env))); v != "" {
		*target = v == "1" || v == "true" || v == "yes" || v == "on"
	}
}

func overrideInt(target *int, env string) {
	if v := strings.TrimSpace(os.Getenv(env)); v != "" {
		if p, err := strconv.Atoi(v); err == nil {
			*target = p
		}
	}
}

func overrideUint64(target *uint64, env string) {
	if v := strings.TrimSpace(os.Getenv(env)); v != "" {
		if p, err := strconv.ParseUint(v, 10, 64); err == nil {
			*target = p
		}
	}
}

func overrideFloat64(target *float64, env string) {
	if v := strings.TrimSpace(os.Getenv(env)); v != "" {
		if p, err := strconv.ParseFloat(v, 64); err == nil {
			*target = p
		}
	}
}

func validateConfig(cfg Config) error {
	if cfg.Listen.Network != "tcp" && cfg.Listen.Network != "unix" {
		return fmt.Errorf("listen.network must be tcp or unix")
	}
	if strings.TrimSpace(cfg.Listen.Address) == "" {
		return fmt.Errorf("listen.address must not be empty")
	}
	if cfg.Pressure.MonitorIntervalMs <= 0 {
		return fmt.Errorf("pressure.monitor_interval_ms must be > 0")
	}
	if cfg.Pressure.SampleTimeoutMs <= 0 {
		return fmt.Errorf("pressure.sample_timeout_ms must be > 0")
	}
	if cfg.Worker.AutoStart || cfg.Worker.AutoRecover {
		if len(cfg.Worker.Command) == 0 {
			return fmt.Errorf("worker.command must be configured when worker lifecycle is enabled")
		}
	}
	if cfg.Worker.HealthFailureLimit <= 0 {
		return fmt.Errorf("worker.health_failure_limit must be > 0")
	}
	if cfg.Worker.StartupHealthPollMs <= 0 {
		return fmt.Errorf("worker.startup_health_poll_ms must be > 0")
	}
	if cfg.Worker.StopTimeoutMs <= 0 {
		return fmt.Errorf("worker.stop_timeout_ms must be > 0")
	}
	if cfg.Backoff.InitialMs <= 0 || cfg.Backoff.MaxMs <= 0 {
		return fmt.Errorf("backoff initial/max must be > 0")
	}
	if cfg.Backoff.MaxMs < cfg.Backoff.InitialMs {
		return fmt.Errorf("backoff.max_ms must be >= backoff.initial_ms")
	}
	if cfg.Backoff.Multiplier < 1.0 {
		return fmt.Errorf("backoff.multiplier must be >= 1")
	}
	if cfg.ControlTimeoutMs <= 0 {
		return fmt.Errorf("control_timeout_ms must be > 0")
	}
	return nil
}

func newSupervisor(cfg Config, logger *slog.Logger) *Supervisor {
	runCtx, runCancel := context.WithCancel(context.Background())
	return &Supervisor{
		cfg:    cfg,
		logger: logger,

		runCtx:    runCtx,
		runCancel: runCancel,

		samplePressureFn: samplePressure,
		checkWorkerHealthFn: func(ctx context.Context, url string, timeout time.Duration) error {
			return checkWorkerHealth(ctx, url, timeout)
		},
	}
}

func samplePressure(ctx context.Context) (PressureSnapshot, error) {
	vm, err := mem.VirtualMemoryWithContext(ctx)
	if err != nil {
		return PressureSnapshot{}, fmt.Errorf("read memory metrics: %w", err)
	}
	cpuVals, err := cpu.PercentWithContext(ctx, 0, false)
	if err != nil {
		return PressureSnapshot{}, fmt.Errorf("read cpu metrics: %w", err)
	}
	cpuPercent := 0.0
	if len(cpuVals) > 0 {
		cpuPercent = cpuVals[0]
	}

	return PressureSnapshot{
		Timestamp:         time.Now().UTC(),
		CPUPercent:        cpuPercent,
		MemoryPercent:     vm.UsedPercent,
		MemoryAvailableMB: vm.Available / 1024 / 1024,
		MemoryTotalMB:     vm.Total / 1024 / 1024,
		Valid:             true,
	}, nil
}

func checkWorkerHealth(ctx context.Context, url string, timeout time.Duration) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("health status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	return nil
}

func (s *Supervisor) start() error {
	if err := s.refreshPressure(context.Background()); err != nil {
		s.logger.Warn("initial pressure sample failed", "error", err.Error())
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/health", s.handleHealth)
	mux.HandleFunc("/v1/metrics", s.handleMetrics)
	mux.HandleFunc("/v1/gates/heavy-load", s.handleHeavyLoadGate)
	mux.HandleFunc("/v1/control/start", s.handleStart)
	mux.HandleFunc("/v1/control/stop", s.handleStop)
	mux.HandleFunc("/v1/control/restart", s.handleRestart)
	mux.HandleFunc("/v1/control/status", s.handleStatus)

	s.httpServer = &http.Server{
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	listener, err := s.createListener()
	if err != nil {
		return err
	}
	s.listener = listener

	s.wg.Add(1)
	go s.monitorLoop()

	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		err := s.httpServer.Serve(listener)
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			s.serveErr.Store(err)
			s.logger.Error("http server exited", "error", err.Error())
		}
	}()

	s.logger.Info(
		"voice sidecar started",
		"network", s.cfg.Listen.Network,
		"address", s.cfg.Listen.Address,
	)

	if s.cfg.Worker.AutoStart {
		ctx, cancel := context.WithTimeout(s.runCtx, s.controlTimeout())
		defer cancel()
		if err := s.startWorker(ctx, "auto_start"); err != nil {
			s.logger.Error("worker auto-start failed", "error", err.Error())
			if s.cfg.Pressure.FailClosed {
				return err
			}
		}
	}

	return nil
}

func (s *Supervisor) createListener() (net.Listener, error) {
	if s.cfg.Listen.Network == "unix" {
		sockPath := s.cfg.Listen.Address
		if err := os.RemoveAll(sockPath); err != nil {
			return nil, fmt.Errorf("remove stale unix socket: %w", err)
		}
		if err := os.MkdirAll(filepath.Dir(sockPath), 0o755); err != nil {
			return nil, fmt.Errorf("create unix socket dir: %w", err)
		}
		ln, err := net.Listen("unix", sockPath)
		if err != nil {
			return nil, fmt.Errorf("listen unix: %w", err)
		}
		if err := os.Chmod(sockPath, 0o660); err != nil {
			s.logger.Warn("could not chmod unix socket", "path", sockPath, "error", err.Error())
		}
		return ln, nil
	}
	ln, err := net.Listen("tcp", s.cfg.Listen.Address)
	if err != nil {
		return nil, fmt.Errorf("listen tcp: %w", err)
	}
	return ln, nil
}

func (s *Supervisor) stop(ctx context.Context) error {
	var stopErr error
	s.stopOnce.Do(func() {
		s.runCancel()

		workerStopCtx, workerStopCancel := context.WithTimeout(context.Background(), time.Duration(s.cfg.Worker.StopTimeoutMs)*time.Millisecond)
		defer workerStopCancel()
		if err := s.stopWorker(workerStopCtx, "sidecar_shutdown"); err != nil {
			s.logger.Warn("worker stop during sidecar shutdown failed", "error", err.Error())
			stopErr = err
		}

		if s.httpServer != nil {
			shutdownCtx, shutdownCancel := context.WithTimeout(ctx, 5*time.Second)
			defer shutdownCancel()
			if err := s.httpServer.Shutdown(shutdownCtx); err != nil {
				s.logger.Warn("http server shutdown failed", "error", err.Error())
				if stopErr == nil {
					stopErr = err
				}
			}
		}

		s.wg.Wait()

		if s.cfg.Listen.Network == "unix" {
			if err := os.Remove(s.cfg.Listen.Address); err != nil && !os.IsNotExist(err) {
				s.logger.Warn("failed to remove unix socket", "path", s.cfg.Listen.Address, "error", err.Error())
			}
		}
	})
	return stopErr
}

func (s *Supervisor) monitorLoop() {
	defer s.wg.Done()
	ticker := time.NewTicker(time.Duration(s.cfg.Pressure.MonitorIntervalMs) * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-s.runCtx.Done():
			return
		case <-ticker.C:
			if err := s.refreshPressure(s.runCtx); err != nil {
				s.logger.Warn("pressure monitor sample failed", "error", err.Error())
			}
			s.applyPressurePolicy()
			s.applyWorkerHealthPolicy()
		}
	}
}

func (s *Supervisor) refreshPressure(ctx context.Context) error {
	timeout := time.Duration(s.cfg.Pressure.SampleTimeoutMs) * time.Millisecond
	sampleCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	snap, err := s.samplePressureFn(sampleCtx)
	if err != nil {
		snap = PressureSnapshot{
			Timestamp: time.Now().UTC(),
			Valid:     false,
			Reason:    err.Error(),
		}
	}

	s.mu.Lock()
	s.pressure = snap
	atomic.AddUint64(&s.metrics.PressureSamples, 1)
	s.mu.Unlock()
	return err
}

func (s *Supervisor) applyPressurePolicy() {
	allowed, reason := s.currentGateState()
	if allowed {
		return
	}
	if !s.cfg.Worker.StopOnPressure {
		return
	}

	s.mu.RLock()
	running := s.workerCmd != nil
	s.mu.RUnlock()
	if !running {
		return
	}

	s.logger.Warn("pressure gate closed; stopping worker", "reason", reason)
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(s.cfg.Worker.StopTimeoutMs)*time.Millisecond)
	defer cancel()
	if err := s.stopWorker(ctx, "pressure_policy"); err != nil {
		s.logger.Warn("worker stop on pressure failed", "error", err.Error())
	}
}

func (s *Supervisor) applyWorkerHealthPolicy() {
	if strings.TrimSpace(s.cfg.Worker.HealthURL) == "" {
		return
	}

	s.mu.RLock()
	running := s.workerCmd != nil
	s.mu.RUnlock()
	if !running {
		return
	}

	timeout := time.Duration(s.cfg.Worker.HealthTimeoutMs) * time.Millisecond
	err := s.checkWorkerHealthFn(s.runCtx, s.cfg.Worker.HealthURL, timeout)

	s.mu.Lock()
	s.lastHealthCheckAt = time.Now().UTC()
	if err == nil {
		s.healthFailureCount = 0
		s.workerHealthCheckPass = true
		s.mu.Unlock()
		return
	}
	
	s.healthFailureCount++
	currentFailures := s.healthFailureCount
	s.workerHealthCheckPass = false
	s.workerLastError = fmt.Sprintf("health check failed: %v", err)
	s.mu.Unlock()

	s.logger.Warn("worker health check failed",
		"failures", currentFailures,
		"limit", s.cfg.Worker.HealthFailureLimit,
		"error", err.Error(),
	)
	if currentFailures < s.cfg.Worker.HealthFailureLimit {
		return
	}

	s.mu.Lock()
	s.healthFailureCount = 0
	s.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), s.controlTimeout())
	defer cancel()
	if err := s.restartWorker(ctx, "health_policy"); err != nil {
		s.logger.Warn("worker restart on health policy failed", "error", err.Error())
	}
}

func (s *Supervisor) controlTimeout() time.Duration {
	return time.Duration(s.cfg.ControlTimeoutMs) * time.Millisecond
}

func (s *Supervisor) currentGateState() (bool, string) {
	s.mu.RLock()
	snap := s.pressure
	s.mu.RUnlock()
	return s.evaluateGate(snap)
}

func (s *Supervisor) evaluateGate(snap PressureSnapshot) (bool, string) {
	if !snap.Valid {
		if s.cfg.Pressure.FailClosed {
			if snap.Reason == "" {
				return false, "pressure metrics unavailable"
			}
			return false, fmt.Sprintf("pressure metrics unavailable: %s", snap.Reason)
		}
		return true, ""
	}

	if snap.MemoryAvailableMB < s.cfg.Pressure.MinAvailableMemoryMB {
		return false, fmt.Sprintf("available memory %dMB below minimum %dMB", snap.MemoryAvailableMB, s.cfg.Pressure.MinAvailableMemoryMB)
	}
	if snap.MemoryPercent > s.cfg.Pressure.MaxMemoryPercent {
		return false, fmt.Sprintf("memory pressure %.1f%% above max %.1f%%", snap.MemoryPercent, s.cfg.Pressure.MaxMemoryPercent)
	}
	if snap.CPUPercent > s.cfg.Pressure.MaxCPUPercent {
		return false, fmt.Sprintf("cpu pressure %.1f%% above max %.1f%%", snap.CPUPercent, s.cfg.Pressure.MaxCPUPercent)
	}
	return true, ""
}

func (s *Supervisor) startWorker(ctx context.Context, reason string) error {
	_, err, _ := s.sf.Do("worker:start", func() (interface{}, error) {
		return nil, s.startWorkerInternal(ctx, reason)
	})
	return err
}

func (s *Supervisor) startWorkerInternal(ctx context.Context, reason string) error {
	s.mu.Lock()
	if s.workerCmd != nil {
		s.mu.Unlock()
		return nil
	}
	if len(s.cfg.Worker.Command) == 0 {
		s.mu.Unlock()
		return fmt.Errorf("worker command is empty")
	}
	if !s.workerBackoffUntil.IsZero() && time.Now().Before(s.workerBackoffUntil) {
		until := s.workerBackoffUntil
		s.mu.Unlock()
		return fmt.Errorf("worker in backoff until %s", until.Format(time.RFC3339Nano))
	}
	s.mu.Unlock()

	allowed, gateReason := s.currentGateState()
	if !allowed {
		s.mu.Lock()
		s.workerLastError = gateReason
		s.mu.Unlock()
		return fmt.Errorf("worker start blocked by pressure gate: %s", gateReason)
	}

	cmd := exec.CommandContext(s.runCtx, s.cfg.Worker.Command[0], s.cfg.Worker.Command[1:]...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if strings.TrimSpace(s.cfg.Worker.WorkingDir) != "" {
		cmd.Dir = s.cfg.Worker.WorkingDir
	}
	if len(s.cfg.Worker.Env) > 0 {
		env := os.Environ()
		for k, v := range s.cfg.Worker.Env {
			env = append(env, fmt.Sprintf("%s=%s", k, v))
		}
		cmd.Env = env
	}

	if err := cmd.Start(); err != nil {
		s.mu.Lock()
		s.workerLastError = err.Error()
		s.mu.Unlock()
		return err
	}

	s.mu.Lock()
	s.workerCmd = cmd
	s.workerPID = cmd.Process.Pid
	s.workerStartedAt = time.Now().UTC()
	s.workerStopRequested = false
	s.workerLastError = ""
	s.workerBackoffUntil = time.Time{}
	s.healthFailureCount = 0
	atomic.AddUint64(&s.metrics.StartRequests, 1)
	pid := s.workerPID
	s.mu.Unlock()

	s.logger.Info("worker started", "pid", pid, "reason", reason, "command", strings.Join(s.cfg.Worker.Command, " "))

	s.wg.Add(1)
	go s.waitWorker(cmd)

	if strings.TrimSpace(s.cfg.Worker.HealthURL) != "" && s.cfg.Worker.StartupHealthWaitMs > 0 {
		if err := s.waitForWorkerHealth(ctx); err != nil {
			s.logger.Warn("worker failed startup health checks", "error", err.Error())
			stopCtx, cancel := context.WithTimeout(context.Background(), time.Duration(s.cfg.Worker.StopTimeoutMs)*time.Millisecond)
			defer cancel()
			_ = s.stopWorker(stopCtx, "startup_health_failed")
			return fmt.Errorf("startup health check failed: %w", err)
		}
	}

	return nil
}

func (s *Supervisor) waitForWorkerHealth(ctx context.Context) error {
	deadline := time.Now().Add(time.Duration(s.cfg.Worker.StartupHealthWaitMs) * time.Millisecond)
	poll := time.Duration(s.cfg.Worker.StartupHealthPollMs) * time.Millisecond
	if poll <= 0 {
		poll = 250 * time.Millisecond
	}

	for {
		if time.Now().After(deadline) {
			return fmt.Errorf("startup health timeout (%dms)", s.cfg.Worker.StartupHealthWaitMs)
		}
		if err := s.checkWorkerHealthFn(ctx, s.cfg.Worker.HealthURL, time.Duration(s.cfg.Worker.HealthTimeoutMs)*time.Millisecond); err == nil {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-s.runCtx.Done():
			return context.Canceled
		case <-time.After(poll):
		}
	}
}

func (s *Supervisor) waitWorker(cmd *exec.Cmd) {
	defer s.wg.Done()
	err := cmd.Wait()

	s.mu.Lock()
	if s.workerCmd != cmd {
		s.mu.Unlock()
		return
	}

	expectedStop := s.workerStopRequested
	exitText := "exited"
	if err != nil {
		exitText = err.Error()
	}
	s.workerLastExit = exitText
	s.workerCmd = nil
	s.workerPID = 0
	s.workerStartedAt = time.Time{}
	s.workerStopRequested = false
	shouldRecover := !expectedStop && s.cfg.Worker.AutoRecover
	if !expectedStop {
		atomic.AddUint64(&s.metrics.CrashCount, 1)
		s.workerRestartCount++
	}
	delay := s.nextBackoffDelayLocked()
	if shouldRecover {
		s.workerBackoffUntil = time.Now().Add(delay)
	} else {
		s.workerBackoffUntil = time.Time{}
		s.backoffExp = 0
	}
	s.mu.Unlock()

	s.logger.Warn("worker exited", "expected", expectedStop, "error", errString(err), "next_backoff_ms", delay.Milliseconds())

	if shouldRecover {
		atomic.AddUint64(&s.metrics.CrashRecoveries, 1)
		go s.recoverAfter(delay, "crash_recovery")
	}
}

func (s *Supervisor) nextBackoffDelayLocked() time.Duration {
	initial := float64(s.cfg.Backoff.InitialMs)
	maxDelay := float64(s.cfg.Backoff.MaxMs)
	exp := math.Pow(s.cfg.Backoff.Multiplier, float64(s.backoffExp))
	delayMs := initial * exp
	if delayMs > maxDelay {
		delayMs = maxDelay
	}
	s.backoffExp++
	return time.Duration(delayMs) * time.Millisecond
}

func (s *Supervisor) recoverAfter(delay time.Duration, reason string) {
	select {
	case <-s.runCtx.Done():
		return
	case <-time.After(delay):
	}

	ctx, cancel := context.WithTimeout(context.Background(), s.controlTimeout())
	defer cancel()
	if err := s.startWorker(ctx, reason); err != nil {
		s.logger.Warn("worker recovery start failed", "error", err.Error())
	}
}

func (s *Supervisor) stopWorker(ctx context.Context, reason string) error {
	_, err, _ := s.sf.Do("worker:stop", func() (interface{}, error) {
		return nil, s.stopWorkerInternal(ctx, reason)
	})
	return err
}

func (s *Supervisor) stopWorkerInternal(ctx context.Context, reason string) error {
	s.mu.Lock()
	cmd := s.workerCmd
	if cmd == nil {
		s.mu.Unlock()
		return nil
	}
	s.workerStopRequested = true
	atomic.AddUint64(&s.metrics.StopRequests, 1)
	s.mu.Unlock()

	s.logger.Info("stopping worker", "pid", cmd.Process.Pid, "reason", reason)
	_ = terminateProcess(cmd.Process)

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			_ = cmd.Process.Kill()
			return fmt.Errorf("worker stop timeout: %w", ctx.Err())
		case <-ticker.C:
			s.mu.RLock()
			stillRunning := s.workerCmd == cmd
			s.mu.RUnlock()
			if !stillRunning {
				return nil
			}
		}
	}
}

func terminateProcess(proc *os.Process) error {
	if proc == nil {
		return nil
	}
	if runtime.GOOS == "windows" {
		return proc.Kill()
	}
	if err := proc.Signal(syscall.SIGTERM); err != nil {
		return proc.Kill()
	}
	return nil
}

func (s *Supervisor) restartWorker(ctx context.Context, reason string) error {
	_, err, _ := s.sf.Do("worker:restart", func() (interface{}, error) {
		atomic.AddUint64(&s.metrics.RestartRequests, 1)
		stopCtx, cancelStop := context.WithTimeout(ctx, time.Duration(s.cfg.Worker.StopTimeoutMs)*time.Millisecond)
		defer cancelStop()
		_ = s.stopWorkerInternal(stopCtx, "restart")
		return nil, s.startWorkerInternal(ctx, reason)
	})
	return err
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func (s *Supervisor) snapshotForResponse() (PressureSnapshot, WorkerStatus, bool, string) {
	s.mu.RLock()
	snap := s.pressure
	status := WorkerStatus{
		State:             s.workerStateLocked(),
		PID:               s.workerPID,
		StartedAt:         s.workerStartedAt,
		RestartCount:      s.workerRestartCount,
		LastExit:          s.workerLastExit,
		LastError:         s.workerLastError,
		BackoffUntil:      s.workerBackoffUntil,
		HealthFailures:    s.healthFailureCount,
		LastHealthCheckAt: s.lastHealthCheckAt,
	}
	s.mu.RUnlock()
	allow, reason := s.evaluateGate(snap)
	return snap, status, allow, reason
}

func (s *Supervisor) workerStateLocked() string {
	if s.workerCmd != nil {
		return "running"
	}
	if !s.workerBackoffUntil.IsZero() && time.Now().Before(s.workerBackoffUntil) {
		return "backoff"
	}
	return "stopped"
}

func (s *Supervisor) handleHealth(w http.ResponseWriter, _ *http.Request) {
	snap, worker, allow, reason := s.snapshotForResponse()
	status := "ok"
	if !allow {
		status = "degraded"
	}

	payload := map[string]interface{}{
		"status": status,
		"time":   time.Now().UTC().Format(time.RFC3339Nano),
		"gate": map[string]interface{}{
			"heavy_load_allowed": allow,
			"reason":             reason,
		},
		"pressure": snap,
		"worker":   worker,
		"listen": map[string]string{
			"network": s.cfg.Listen.Network,
			"address": s.cfg.Listen.Address,
		},
	}
	writeJSON(w, http.StatusOK, payload)
}

func (s *Supervisor) handleStatus(w http.ResponseWriter, _ *http.Request) {
	snap, worker, allow, reason := s.snapshotForResponse()
	payload := map[string]interface{}{
		"gate": map[string]interface{}{
			"heavy_load_allowed": allow,
			"reason":             reason,
		},
		"pressure": snap,
		"worker":   worker,
	}
	writeJSON(w, http.StatusOK, payload)
}

func (s *Supervisor) handleHeavyLoadGate(w http.ResponseWriter, _ *http.Request) {
	snap, _, allow, reason := s.snapshotForResponse()
	payload := map[string]interface{}{
		"allowed": allow,
		"reason":  reason,
		"pressure": snap,
	}
	status := http.StatusOK
	if !allow {
		status = http.StatusTooManyRequests
	}
	writeJSON(w, status, payload)
}

func (s *Supervisor) handleStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	req := parseControlRequest(r)
	ctx, cancel := context.WithTimeout(r.Context(), s.controlTimeout())
	defer cancel()
	if err := s.startWorker(ctx, firstNonEmpty(req.Reason, "api_start")); err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "started"})
}

func (s *Supervisor) handleStop(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	req := parseControlRequest(r)
	ctx, cancel := context.WithTimeout(r.Context(), s.controlTimeout())
	defer cancel()
	if err := s.stopWorker(ctx, firstNonEmpty(req.Reason, "api_stop")); err != nil {
		writeJSON(w, http.StatusGatewayTimeout, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "stopped"})
}

func (s *Supervisor) handleRestart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	req := parseControlRequest(r)
	ctx, cancel := context.WithTimeout(r.Context(), s.controlTimeout())
	defer cancel()
	if err := s.restartWorker(ctx, firstNonEmpty(req.Reason, "api_restart")); err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "restarted"})
}

func (s *Supervisor) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	snap, worker, allow, _ := s.snapshotForResponse()

	buf := bytes.NewBuffer(nil)
	fmt.Fprintf(buf, "# HELP jarvis_voice_sidecar_gate_open Whether heavy-load gate is open (1=true).\n")
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_gate_open gauge\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_gate_open %d\n", boolToInt(allow))

	fmt.Fprintf(buf, "# HELP jarvis_voice_sidecar_worker_running Whether worker is running (1=true).\n")
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_worker_running gauge\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_worker_running %d\n", boolToInt(worker.State == "running"))

	fmt.Fprintf(buf, "# HELP jarvis_voice_sidecar_cpu_percent Latest CPU percent.\n")
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_cpu_percent gauge\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_cpu_percent %.4f\n", snap.CPUPercent)

	fmt.Fprintf(buf, "# HELP jarvis_voice_sidecar_memory_percent Latest memory percent.\n")
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_memory_percent gauge\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_memory_percent %.4f\n", snap.MemoryPercent)

	fmt.Fprintf(buf, "# HELP jarvis_voice_sidecar_memory_available_mb Latest available memory in MB.\n")
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_memory_available_mb gauge\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_memory_available_mb %d\n", snap.MemoryAvailableMB)

	startRequests := atomic.LoadUint64(&s.metrics.StartRequests)
	stopRequests := atomic.LoadUint64(&s.metrics.StopRequests)
	restartRequests := atomic.LoadUint64(&s.metrics.RestartRequests)
	crashRecoveries := atomic.LoadUint64(&s.metrics.CrashRecoveries)
	crashes := atomic.LoadUint64(&s.metrics.CrashCount)
	pressureSamples := atomic.LoadUint64(&s.metrics.PressureSamples)

	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_start_requests_total counter\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_start_requests_total %d\n", startRequests)
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_stop_requests_total counter\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_stop_requests_total %d\n", stopRequests)
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_restart_requests_total counter\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_restart_requests_total %d\n", restartRequests)
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_crash_recoveries_total counter\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_crash_recoveries_total %d\n", crashRecoveries)
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_crash_total counter\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_crash_total %d\n", crashes)
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_pressure_samples_total counter\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_pressure_samples_total %d\n", pressureSamples)

	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(buf.Bytes())
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func parseControlRequest(r *http.Request) ControlRequest {
	if r.Body == nil {
		return ControlRequest{}
	}
	defer r.Body.Close()
	buf, err := io.ReadAll(io.LimitReader(r.Body, 2048))
	if err != nil || len(bytes.TrimSpace(buf)) == 0 {
		return ControlRequest{}
	}
	var req ControlRequest
	if err := json.Unmarshal(buf, &req); err != nil {
		return ControlRequest{}
	}
	return req
}

func writeJSON(w http.ResponseWriter, status int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func main() {
	configPath := flag.String("config", "", "Path to sidecar config (yaml/json)")
	flag.Parse()

	cfg, err := loadConfig(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid config: %v\n", err)
		os.Exit(2)
	}

	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	sidecar := newSupervisor(cfg, logger)

	if err := sidecar.start(); err != nil {
		logger.Error("failed to start sidecar", "error", err.Error())
		os.Exit(1)
	}

	signalCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	<-signalCtx.Done()

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := sidecar.stop(shutdownCtx); err != nil {
		logger.Warn("sidecar shutdown completed with errors", "error", err.Error())
	}
}
