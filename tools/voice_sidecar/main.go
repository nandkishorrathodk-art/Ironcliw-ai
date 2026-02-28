package main

import (
	"bufio"
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
	Listen   ListenConfig   `json:"listen" yaml:"listen"`
	Poll     PollConfig     `json:"poll" yaml:"poll"`
	Python   PythonConfig   `json:"python" yaml:"python"`
	Pressure PressureConfig `json:"pressure" yaml:"pressure"`
	Advisory AdvisoryConfig `json:"advisory" yaml:"advisory"`
	Backoff  BackoffConfig  `json:"backoff" yaml:"backoff"`
}

type ListenConfig struct {
	Network string `json:"network" yaml:"network"`
	Address string `json:"address" yaml:"address"`
}

type PollConfig struct {
	IntervalMs      int `json:"interval_ms" yaml:"interval_ms"`
	RequestTimeoutMs int `json:"request_timeout_ms" yaml:"request_timeout_ms"`
}

type PythonConfig struct {
	Transport      string `json:"transport" yaml:"transport"`
	StatusURL      string `json:"status_url" yaml:"status_url"`
	HealthURL      string `json:"health_url" yaml:"health_url"`
	UnixSocketPath string `json:"unix_socket_path" yaml:"unix_socket_path"`
}

type PressureConfig struct {
	FailClosed           bool    `json:"fail_closed" yaml:"fail_closed"`
	MinAvailableMemoryMB uint64  `json:"min_available_memory_mb" yaml:"min_available_memory_mb"`
	MaxMemoryPercent     float64 `json:"max_memory_percent" yaml:"max_memory_percent"`
	MaxCPUPercent        float64 `json:"max_cpu_percent" yaml:"max_cpu_percent"`
}

type AdvisoryConfig struct {
	RecoveryStuckSeconds     int `json:"recovery_stuck_seconds" yaml:"recovery_stuck_seconds"`
	ModeOscillationWindowSec int `json:"mode_oscillation_window_sec" yaml:"mode_oscillation_window_sec"`
	ModeOscillationLimit     int `json:"mode_oscillation_limit" yaml:"mode_oscillation_limit"`
}

type BackoffConfig struct {
	InitialMs  int     `json:"initial_ms" yaml:"initial_ms"`
	MaxMs      int     `json:"max_ms" yaml:"max_ms"`
	Multiplier float64 `json:"multiplier" yaml:"multiplier"`
}

type PressureSnapshot struct {
	Timestamp         time.Time `json:"timestamp"`
	CPUPercent        float64   `json:"cpu_percent"`
	MemoryPercent     float64   `json:"memory_percent"`
	MemoryAvailableMB uint64    `json:"memory_available_mb"`
	MemoryTotalMB     uint64    `json:"memory_total_mb"`
	Valid             bool      `json:"valid"`
	Reason            string    `json:"reason,omitempty"`
}

type AdvisorySignal struct {
	Code      string    `json:"code"`
	Severity  string    `json:"severity"`
	Message   string    `json:"message"`
	Triggered time.Time `json:"triggered"`
}

type ObservedState struct {
	Timestamp         time.Time        `json:"timestamp"`
	PythonReachable   bool             `json:"python_reachable"`
	DesiredMode       string           `json:"desired_mode"`
	EffectiveMode     string           `json:"effective_mode"`
	MemoryTier        string           `json:"memory_tier"`
	RecoveryState     string           `json:"recovery_state"`
	LocalCircuitState string           `json:"local_circuit_state"`
	Pressure          PressureSnapshot `json:"pressure"`
	Advisories        []AdvisorySignal `json:"advisories"`
	RawStatus         map[string]any   `json:"raw_status,omitempty"`
}

type Metrics struct {
	PollErrors uint64
	PollSuccess uint64
}

type Observer struct {
	cfg    Config
	logger *slog.Logger

	httpServer *http.Server
	listener   net.Listener

	runCtx    context.Context
	runCancel context.CancelFunc
	stopOnce  sync.Once
	wg        sync.WaitGroup

	sf singleflight.Group

	mu         sync.RWMutex
	state      ObservedState
	failures   int
	backoffTil time.Time
	metrics    Metrics

	lastEffectiveMode      string
	modeChangeTimestamps   []time.Time
	recoveryActiveSince    time.Time
	recoveryStuckTriggered bool
}

func defaultConfig() Config {
	home, _ := os.UserHomeDir()
	defaultSocket := filepath.Join(home, ".jarvis", "locks", "kernel.sock")
	return Config{
		Listen: ListenConfig{
			Network: "tcp",
			Address: "127.0.0.1:9860",
		},
		Poll: PollConfig{
			IntervalMs:      1000,
			RequestTimeoutMs: 900,
		},
		Python: PythonConfig{
			Transport:      "unix",
			StatusURL:      "http://127.0.0.1:8000/supervisor/status",
			HealthURL:      "http://127.0.0.1:8000/supervisor/health",
			UnixSocketPath: defaultSocket,
		},
		Pressure: PressureConfig{
			FailClosed:           true,
			MinAvailableMemoryMB: 2048,
			MaxMemoryPercent:     90.0,
			MaxCPUPercent:        95.0,
		},
		Advisory: AdvisoryConfig{
			RecoveryStuckSeconds:     180,
			ModeOscillationWindowSec: 120,
			ModeOscillationLimit:     6,
		},
		Backoff: BackoffConfig{
			InitialMs:  500,
			MaxMs:      10000,
			Multiplier: 2.0,
		},
	}
}

func loadConfig(path string) (Config, error) {
	cfg := defaultConfig()
	if strings.TrimSpace(path) != "" {
		buf, err := os.ReadFile(path)
		if err != nil {
			return cfg, fmt.Errorf("read config: %w", err)
		}
		if err := yaml.Unmarshal(buf, &cfg); err != nil {
			return cfg, fmt.Errorf("parse yaml config: %w", err)
		}
	}
	applyEnvOverrides(&cfg)
	cfg.Python.UnixSocketPath = expandHomePath(cfg.Python.UnixSocketPath)
	if err := validateConfig(cfg); err != nil {
		return cfg, err
	}
	return cfg, nil
}

func expandHomePath(path string) string {
	trimmed := strings.TrimSpace(path)
	if trimmed == "~" {
		home, err := os.UserHomeDir()
		if err == nil {
			return home
		}
		return trimmed
	}
	if strings.HasPrefix(trimmed, "~/") {
		home, err := os.UserHomeDir()
		if err == nil {
			return filepath.Join(home, trimmed[2:])
		}
	}
	return trimmed
}

func applyEnvOverrides(cfg *Config) {
	overrideString(&cfg.Listen.Network, "Ironcliw_VOICE_SIDECAR_NETWORK")
	overrideString(&cfg.Listen.Address, "Ironcliw_VOICE_SIDECAR_ADDRESS")
	overrideInt(&cfg.Poll.IntervalMs, "Ironcliw_VOICE_SIDECAR_POLL_INTERVAL_MS")
	overrideInt(&cfg.Poll.RequestTimeoutMs, "Ironcliw_VOICE_SIDECAR_REQUEST_TIMEOUT_MS")
	overrideString(&cfg.Python.Transport, "Ironcliw_VOICE_SIDECAR_PYTHON_TRANSPORT")
	overrideString(&cfg.Python.StatusURL, "Ironcliw_VOICE_SIDECAR_STATUS_URL")
	overrideString(&cfg.Python.HealthURL, "Ironcliw_VOICE_SIDECAR_HEALTH_URL")
	overrideString(&cfg.Python.UnixSocketPath, "Ironcliw_VOICE_SIDECAR_SOCKET")
	overrideBool(&cfg.Pressure.FailClosed, "Ironcliw_VOICE_SIDECAR_FAIL_CLOSED")
	overrideUint64(&cfg.Pressure.MinAvailableMemoryMB, "Ironcliw_VOICE_SIDECAR_MIN_MEM_MB")
	overrideFloat64(&cfg.Pressure.MaxMemoryPercent, "Ironcliw_VOICE_SIDECAR_MAX_MEM_PERCENT")
	overrideFloat64(&cfg.Pressure.MaxCPUPercent, "Ironcliw_VOICE_SIDECAR_MAX_CPU_PERCENT")
	overrideInt(&cfg.Advisory.RecoveryStuckSeconds, "Ironcliw_VOICE_SIDECAR_RECOVERY_STUCK_SECONDS")
	overrideInt(&cfg.Advisory.ModeOscillationWindowSec, "Ironcliw_VOICE_SIDECAR_MODE_OSCILLATION_WINDOW_SEC")
	overrideInt(&cfg.Advisory.ModeOscillationLimit, "Ironcliw_VOICE_SIDECAR_MODE_OSCILLATION_LIMIT")
	overrideInt(&cfg.Backoff.InitialMs, "Ironcliw_VOICE_SIDECAR_BACKOFF_INITIAL_MS")
	overrideInt(&cfg.Backoff.MaxMs, "Ironcliw_VOICE_SIDECAR_BACKOFF_MAX_MS")
	overrideFloat64(&cfg.Backoff.Multiplier, "Ironcliw_VOICE_SIDECAR_BACKOFF_MULTIPLIER")
}

func validateConfig(cfg Config) error {
	if strings.TrimSpace(cfg.Listen.Network) == "" {
		return errors.New("listen.network must be set")
	}
	if strings.TrimSpace(cfg.Listen.Address) == "" {
		return errors.New("listen.address must be set")
	}
	if cfg.Poll.IntervalMs < 250 {
		return errors.New("poll.interval_ms must be >= 250")
	}
	if cfg.Poll.RequestTimeoutMs < 200 {
		return errors.New("poll.request_timeout_ms must be >= 200")
	}
	cfg.Python.Transport = strings.ToLower(strings.TrimSpace(cfg.Python.Transport))
	if cfg.Python.Transport != "http" && cfg.Python.Transport != "unix" {
		return errors.New("python.transport must be http or unix")
	}
	if cfg.Python.Transport == "http" && strings.TrimSpace(cfg.Python.StatusURL) == "" {
		return errors.New("python.status_url must be set for http transport")
	}
	if cfg.Python.Transport == "unix" && strings.TrimSpace(cfg.Python.UnixSocketPath) == "" {
		return errors.New("python.unix_socket_path must be set for unix transport")
	}
	if cfg.Backoff.InitialMs <= 0 || cfg.Backoff.MaxMs <= 0 {
		return errors.New("backoff initial/max must be > 0")
	}
	if cfg.Backoff.Multiplier < 1.0 {
		return errors.New("backoff.multiplier must be >= 1")
	}
	if cfg.Advisory.RecoveryStuckSeconds < 30 {
		return errors.New("advisory.recovery_stuck_seconds must be >= 30")
	}
	if cfg.Advisory.ModeOscillationWindowSec < 30 {
		return errors.New("advisory.mode_oscillation_window_sec must be >= 30")
	}
	if cfg.Advisory.ModeOscillationLimit < 2 {
		return errors.New("advisory.mode_oscillation_limit must be >= 2")
	}
	return nil
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
		if parsed, err := strconv.Atoi(v); err == nil {
			*target = parsed
		}
	}
}

func overrideUint64(target *uint64, env string) {
	if v := strings.TrimSpace(os.Getenv(env)); v != "" {
		if parsed, err := strconv.ParseUint(v, 10, 64); err == nil {
			*target = parsed
		}
	}
}

func overrideFloat64(target *float64, env string) {
	if v := strings.TrimSpace(os.Getenv(env)); v != "" {
		if parsed, err := strconv.ParseFloat(v, 64); err == nil {
			*target = parsed
		}
	}
}

func samplePressure(ctx context.Context) (PressureSnapshot, error) {
	totalTimeout := time.Until(deadlineFromContext(ctx, 900*time.Millisecond))
	if totalTimeout <= 0 {
		return PressureSnapshot{}, context.DeadlineExceeded
	}
	vmStat, err := mem.VirtualMemoryWithContext(ctx)
	if err != nil {
		return PressureSnapshot{}, fmt.Errorf("read memory metrics: %w", err)
	}
	cpuSamples, err := cpu.PercentWithContext(ctx, 0, false)
	if err != nil {
		return PressureSnapshot{}, fmt.Errorf("read cpu metrics: %w", err)
	}
	cpuPct := 0.0
	if len(cpuSamples) > 0 {
		cpuPct = cpuSamples[0]
	}
	return PressureSnapshot{
		Timestamp:         time.Now().UTC(),
		CPUPercent:        cpuPct,
		MemoryPercent:     vmStat.UsedPercent,
		MemoryAvailableMB: vmStat.Available / 1024 / 1024,
		MemoryTotalMB:     vmStat.Total / 1024 / 1024,
		Valid:             true,
		Reason:            fmt.Sprintf("sample_timeout=%s", totalTimeout),
	}, nil
}

func deadlineFromContext(ctx context.Context, fallback time.Duration) time.Time {
	if deadline, ok := ctx.Deadline(); ok {
		return deadline
	}
	return time.Now().Add(fallback)
}

func newObserver(cfg Config, logger *slog.Logger) *Observer {
	return &Observer{cfg: cfg, logger: logger}
}

func (o *Observer) start() error {
	o.runCtx, o.runCancel = context.WithCancel(context.Background())

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", o.handleHealthz)
	mux.HandleFunc("/metrics", o.handleMetrics)
	mux.HandleFunc("/v1/health", o.handleHealth)
	mux.HandleFunc("/v1/observer/state", o.handleObserverState)
	mux.HandleFunc("/v1/gates/heavy-load", o.handleHeavyLoadGate)

	o.httpServer = &http.Server{Handler: mux}
	ln, err := net.Listen(o.cfg.Listen.Network, o.cfg.Listen.Address)
	if err != nil {
		return fmt.Errorf("listen %s %s: %w", o.cfg.Listen.Network, o.cfg.Listen.Address, err)
	}
	o.listener = ln

	o.wg.Add(1)
	go func() {
		defer o.wg.Done()
		err := o.httpServer.Serve(ln)
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			o.logger.Error("http serve failed", "error", err.Error())
		}
	}()

	o.wg.Add(1)
	go func() {
		defer o.wg.Done()
		o.pollLoop()
	}()

	o.logger.Info("voice sidecar observer started",
		"listen", o.cfg.Listen.Address,
		"transport", o.cfg.Python.Transport,
	)
	return nil
}

func (o *Observer) stop(ctx context.Context) error {
	var stopErr error
	o.stopOnce.Do(func() {
		if o.runCancel != nil {
			o.runCancel()
		}
		if o.httpServer != nil {
			if err := o.httpServer.Shutdown(ctx); err != nil && !errors.Is(err, context.Canceled) {
				stopErr = err
			}
		}
		o.wg.Wait()
	})
	return stopErr
}

func (o *Observer) pollLoop() {
	interval := time.Duration(o.cfg.Poll.IntervalMs) * time.Millisecond
	for {
		select {
		case <-o.runCtx.Done():
			return
		default:
		}

		now := time.Now()
		o.mu.RLock()
		backoffTil := o.backoffTil
		o.mu.RUnlock()
		if backoffTil.After(now) {
			timer := time.NewTimer(backoffTil.Sub(now))
			select {
			case <-o.runCtx.Done():
				timer.Stop()
				return
			case <-timer.C:
			}
		}

		obsErr := o.observeOnce()
		if obsErr != nil {
			atomic.AddUint64(&o.metrics.PollErrors, 1)
			o.applyPollFailure(obsErr)
		} else {
			atomic.AddUint64(&o.metrics.PollSuccess, 1)
			o.resetPollFailure()
		}

		timer := time.NewTimer(interval)
		select {
		case <-o.runCtx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
	}
}

func (o *Observer) observeOnce() error {
	_, err, _ := o.sf.Do("observe", func() (interface{}, error) {
		ctx, cancel := context.WithTimeout(o.runCtx, time.Duration(o.cfg.Poll.RequestTimeoutMs)*time.Millisecond)
		defer cancel()

		pressure, pressureErr := samplePressure(ctx)
		if pressureErr != nil {
			pressure = PressureSnapshot{Timestamp: time.Now().UTC(), Valid: false, Reason: pressureErr.Error()}
		}

		status, fetchErr := o.fetchPythonStatus(ctx)
		if fetchErr != nil {
			o.updateObservedState(ObservedState{
				Timestamp:       time.Now().UTC(),
				PythonReachable: false,
				DesiredMode:     "unknown",
				EffectiveMode:   "unknown",
				MemoryTier:      "unknown",
				RecoveryState:   "idle",
				Pressure:        pressure,
				Advisories:      o.computeAdvisories("unknown", "unknown", "idle", pressure),
			})
			return nil, fetchErr
		}

		desired, effective, memTier, recovery, localCircuit := extractSignals(status, pressure)
		state := ObservedState{
			Timestamp:         time.Now().UTC(),
			PythonReachable:   true,
			DesiredMode:       desired,
			EffectiveMode:     effective,
			MemoryTier:        memTier,
			RecoveryState:     recovery,
			LocalCircuitState: localCircuit,
			Pressure:          pressure,
			RawStatus:         status,
		}
		state.Advisories = o.computeAdvisories(desired, effective, recovery, pressure)
		o.updateObservedState(state)
		return nil, nil
	})
	return err
}

func (o *Observer) fetchPythonStatus(ctx context.Context) (map[string]any, error) {
	if strings.EqualFold(o.cfg.Python.Transport, "unix") {
		return fetchStatusViaUnix(ctx, o.cfg.Python.UnixSocketPath)
	}
	return fetchStatusViaHTTP(ctx, o.cfg.Python.StatusURL)
}

func fetchStatusViaHTTP(ctx context.Context, url string) (map[string]any, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("status url %s -> %d: %s", url, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var payload map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, err
	}
	if wrapped, ok := payload["result"].(map[string]any); ok {
		return wrapped, nil
	}
	return payload, nil
}

func fetchStatusViaUnix(ctx context.Context, socketPath string) (map[string]any, error) {
	dialer := net.Dialer{Timeout: 800 * time.Millisecond}
	conn, err := dialer.DialContext(ctx, "unix", socketPath)
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	_ = conn.SetDeadline(deadlineFromContext(ctx, 800*time.Millisecond))

	req := map[string]any{"command": "status", "args": map[string]any{}}
	buf, _ := json.Marshal(req)
	if _, err := conn.Write(append(buf, '\n')); err != nil {
		return nil, err
	}

	line, err := bufio.NewReader(conn).ReadBytes('\n')
	if err != nil {
		return nil, err
	}

	var envelope map[string]any
	if err := json.Unmarshal(line, &envelope); err != nil {
		return nil, err
	}
	if success, _ := envelope["success"].(bool); !success {
		return nil, fmt.Errorf("ipc status failed: %v", envelope["error"])
	}
	result, ok := envelope["result"].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("ipc status missing result")
	}
	return result, nil
}

func extractSignals(status map[string]any, pressure PressureSnapshot) (string, string, string, string, string) {
	startup := mapGet(status, "startup_modes")
	desired := stringOrDefault(startup["desired_mode"], "unknown")
	effective := stringOrDefault(startup["effective_mode"], "unknown")

	pressureSignal := mapGet(status, "memory_pressure_signal")
	memoryTier := stringOrDefault(pressureSignal["status"], "unknown")
	if memoryTier == "unknown" {
		memoryTier = deriveMemoryTier(pressure.MemoryPercent)
	}
	recoveryState := stringOrDefault(pressureSignal["recovery_state"], "idle")
	localCircuit := stringOrDefault(pressureSignal["local_circuit_state"], "unknown")
	return desired, effective, memoryTier, recoveryState, localCircuit
}

func mapGet(container map[string]any, key string) map[string]any {
	if container == nil {
		return map[string]any{}
	}
	v, ok := container[key].(map[string]any)
	if !ok {
		return map[string]any{}
	}
	return v
}

func stringOrDefault(v any, fallback string) string {
	s, ok := v.(string)
	if !ok || strings.TrimSpace(s) == "" {
		return fallback
	}
	return s
}

func deriveMemoryTier(memoryPercent float64) string {
	switch {
	case memoryPercent >= 90:
		return "critical"
	case memoryPercent >= 75:
		return "elevated"
	default:
		return "normal"
	}
}

func (o *Observer) computeAdvisories(desired, effective, recovery string, pressure PressureSnapshot) []AdvisorySignal {
	now := time.Now().UTC()
	advisories := make([]AdvisorySignal, 0, 2)

	o.mu.Lock()
	if o.lastEffectiveMode != "" && effective != "" && effective != "unknown" && effective != o.lastEffectiveMode {
		o.modeChangeTimestamps = append(o.modeChangeTimestamps, now)
	}
	if effective != "" && effective != "unknown" {
		o.lastEffectiveMode = effective
	}
	windowCutoff := now.Add(-time.Duration(o.cfg.Advisory.ModeOscillationWindowSec) * time.Second)
	trimmed := make([]time.Time, 0, len(o.modeChangeTimestamps))
	for _, ts := range o.modeChangeTimestamps {
		if ts.After(windowCutoff) {
			trimmed = append(trimmed, ts)
		}
	}
	o.modeChangeTimestamps = trimmed
	oscillationRisk := len(trimmed) >= o.cfg.Advisory.ModeOscillationLimit

	recoveryActive := recovery == "reloading" || recovery == "armed" || recovery == "cooldown"
	if recoveryActive {
		if o.recoveryActiveSince.IsZero() {
			o.recoveryActiveSince = now
			o.recoveryStuckTriggered = false
		}
	} else {
		o.recoveryActiveSince = time.Time{}
		o.recoveryStuckTriggered = false
	}
	recoveryStuck := false
	if !o.recoveryActiveSince.IsZero() {
		elapsed := now.Sub(o.recoveryActiveSince)
		if elapsed >= time.Duration(o.cfg.Advisory.RecoveryStuckSeconds)*time.Second {
			recoveryStuck = true
		}
	}
	o.mu.Unlock()

	if recoveryStuck {
		advisories = append(advisories, AdvisorySignal{
			Code:      "recovery_stuck",
			Severity:  "warning",
			Message:   "Recovery state has remained active beyond advisory threshold",
			Triggered: now,
		})
	}
	if oscillationRisk {
		advisories = append(advisories, AdvisorySignal{
			Code:      "mode_oscillation_risk",
			Severity:  "warning",
			Message:   fmt.Sprintf("Effective mode changed too frequently (desired=%s effective=%s)", desired, effective),
			Triggered: now,
		})
	}
	if !pressure.Valid && o.cfg.Pressure.FailClosed {
		advisories = append(advisories, AdvisorySignal{
			Code:      "pressure_metrics_unavailable",
			Severity:  "critical",
			Message:   "Pressure metrics unavailable under fail-closed policy",
			Triggered: now,
		})
	}
	return advisories
}

func (o *Observer) updateObservedState(state ObservedState) {
	o.mu.Lock()
	o.state = state
	o.mu.Unlock()
}

func (o *Observer) applyPollFailure(err error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.failures++
	delayMs := float64(o.cfg.Backoff.InitialMs) * math.Pow(o.cfg.Backoff.Multiplier, float64(max(0, o.failures-1)))
	delayMs = math.Min(delayMs, float64(o.cfg.Backoff.MaxMs))
	o.backoffTil = time.Now().Add(time.Duration(delayMs) * time.Millisecond)
	o.logger.Warn("python observation failed",
		"error", err.Error(),
		"failure_count", o.failures,
		"backoff_ms", int(delayMs),
	)
}

func (o *Observer) resetPollFailure() {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.failures = 0
	o.backoffTil = time.Time{}
}

func (o *Observer) snapshot() ObservedState {
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.state
}

func (o *Observer) evaluateHeavyLoadGate(state ObservedState) (bool, string) {
	snap := state.Pressure
	if !snap.Valid {
		if o.cfg.Pressure.FailClosed {
			return false, firstNonEmpty(snap.Reason, "pressure metrics unavailable")
		}
		return true, "pressure metrics unavailable (fail-open)"
	}
	if snap.MemoryAvailableMB < o.cfg.Pressure.MinAvailableMemoryMB {
		return false, fmt.Sprintf("memory available %dMB < %dMB", snap.MemoryAvailableMB, o.cfg.Pressure.MinAvailableMemoryMB)
	}
	if snap.MemoryPercent > o.cfg.Pressure.MaxMemoryPercent {
		return false, fmt.Sprintf("memory percent %.1f > %.1f", snap.MemoryPercent, o.cfg.Pressure.MaxMemoryPercent)
	}
	if snap.CPUPercent > o.cfg.Pressure.MaxCPUPercent {
		return false, fmt.Sprintf("cpu percent %.1f > %.1f", snap.CPUPercent, o.cfg.Pressure.MaxCPUPercent)
	}
	return true, "within pressure limits"
}

func (o *Observer) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	state := o.snapshot()
	status := "ok"
	if !state.PythonReachable {
		status = "degraded"
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status":           status,
		"python_reachable": state.PythonReachable,
		"timestamp":        time.Now().UTC(),
	})
}

func (o *Observer) handleHealth(w http.ResponseWriter, _ *http.Request) {
	state := o.snapshot()
	allowed, reason := o.evaluateHeavyLoadGate(state)
	writeJSON(w, http.StatusOK, map[string]any{
		"status":            "ok",
		"python_reachable":  state.PythonReachable,
		"desired_mode":      state.DesiredMode,
		"effective_mode":    state.EffectiveMode,
		"memory_tier":       state.MemoryTier,
		"recovery_state":    state.RecoveryState,
		"local_circuit":     state.LocalCircuitState,
		"heavy_load_allowed": allowed,
		"heavy_load_reason":  reason,
		"advisories":         state.Advisories,
		"timestamp":          state.Timestamp,
	})
}

func (o *Observer) handleObserverState(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, o.snapshot())
}

func (o *Observer) handleHeavyLoadGate(w http.ResponseWriter, _ *http.Request) {
	state := o.snapshot()
	allowed, reason := o.evaluateHeavyLoadGate(state)
	statusCode := http.StatusOK
	if !allowed {
		statusCode = http.StatusTooManyRequests
	}
	writeJSON(w, statusCode, map[string]any{
		"allowed": allowed,
		"reason":  reason,
		"pressure": state.Pressure,
	})
}

func (o *Observer) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	state := o.snapshot()
	allowed, _ := o.evaluateHeavyLoadGate(state)
	pollErrors := atomic.LoadUint64(&o.metrics.PollErrors)
	pollSuccess := atomic.LoadUint64(&o.metrics.PollSuccess)

	buf := &strings.Builder{}
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_python_reachable gauge\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_python_reachable %d\n", boolToInt(state.PythonReachable))
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_heavy_load_allowed gauge\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_heavy_load_allowed %d\n", boolToInt(allowed))
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_pressure_memory_percent gauge\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_pressure_memory_percent %.4f\n", state.Pressure.MemoryPercent)
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_pressure_cpu_percent gauge\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_pressure_cpu_percent %.4f\n", state.Pressure.CPUPercent)
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_pressure_memory_available_mb gauge\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_pressure_memory_available_mb %d\n", state.Pressure.MemoryAvailableMB)
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_poll_errors_total counter\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_poll_errors_total %d\n", pollErrors)
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_poll_success_total counter\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_poll_success_total %d\n", pollSuccess)
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_mode_info gauge\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_mode_info{desired=\"%s\",effective=\"%s\"} 1\n",
		escapeLabelValue(state.DesiredMode),
		escapeLabelValue(state.EffectiveMode),
	)
	fmt.Fprintf(buf, "# TYPE jarvis_voice_sidecar_recovery_state_info gauge\n")
	fmt.Fprintf(buf, "jarvis_voice_sidecar_recovery_state_info{state=\"%s\",local_circuit=\"%s\"} 1\n",
		escapeLabelValue(state.RecoveryState),
		escapeLabelValue(state.LocalCircuitState),
	)
	for _, advisory := range state.Advisories {
		fmt.Fprintf(buf,
			"jarvis_voice_sidecar_advisory_active{code=\"%s\",severity=\"%s\"} 1\n",
			escapeLabelValue(advisory.Code),
			escapeLabelValue(advisory.Severity),
		)
	}

	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	_, _ = w.Write([]byte(buf.String()))
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

func firstNonEmpty(v, fallback string) string {
	if strings.TrimSpace(v) == "" {
		return fallback
	}
	return v
}

func escapeLabelValue(v string) string {
	r := strings.ReplaceAll(v, "\\", "\\\\")
	r = strings.ReplaceAll(r, "\"", "\\\"")
	r = strings.ReplaceAll(r, "\n", "\\n")
	return r
}

func writeJSON(w http.ResponseWriter, statusCode int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	_ = json.NewEncoder(w).Encode(payload)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	var configPath string
	flag.StringVar(&configPath, "config", "", "Path to sidecar YAML config")
	flag.Parse()

	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	cfg, err := loadConfig(configPath)
	if err != nil {
		logger.Error("invalid configuration", "error", err.Error())
		os.Exit(2)
	}

	obs := newObserver(cfg, logger)
	if err := obs.start(); err != nil {
		logger.Error("failed to start observer", "error", err.Error())
		os.Exit(1)
	}

	signalCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	<-signalCtx.Done()

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := obs.stop(shutdownCtx); err != nil {
		logger.Error("observer shutdown error", "error", err.Error())
	}

	logger.Info("voice sidecar observer stopped", "go_version", runtime.Version())
}
