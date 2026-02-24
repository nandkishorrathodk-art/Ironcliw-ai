package main

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func testLogger() *slog.Logger {
	return slog.New(slog.NewJSONHandler(io.Discard, nil))
}

func TestExtractSignalsPrefersPythonStatus(t *testing.T) {
	status := map[string]any{
		"startup_modes": map[string]any{
			"desired_mode":   "cloud_first",
			"effective_mode": "sequential",
		},
		"memory_pressure_signal": map[string]any{
			"status":              "elevated",
			"recovery_state":      "cooldown",
			"local_circuit_state": "open",
		},
	}

	desired, effective, tier, recovery, circuit := extractSignals(status, PressureSnapshot{MemoryPercent: 15})
	if desired != "cloud_first" {
		t.Fatalf("desired mismatch: %s", desired)
	}
	if effective != "sequential" {
		t.Fatalf("effective mismatch: %s", effective)
	}
	if tier != "elevated" {
		t.Fatalf("tier mismatch: %s", tier)
	}
	if recovery != "cooldown" {
		t.Fatalf("recovery mismatch: %s", recovery)
	}
	if circuit != "open" {
		t.Fatalf("circuit mismatch: %s", circuit)
	}
}

func TestHeavyLoadGateFailClosed(t *testing.T) {
	cfg := defaultConfig()
	cfg.Pressure.FailClosed = true
	obs := newObserver(cfg, testLogger())

	allowed, reason := obs.evaluateHeavyLoadGate(ObservedState{
		Pressure: PressureSnapshot{Valid: false, Reason: "metrics unavailable"},
	})
	if allowed {
		t.Fatalf("expected fail-closed gate to deny")
	}
	if reason == "" {
		t.Fatalf("expected gate reason")
	}
}

func TestModeOscillationAdvisory(t *testing.T) {
	cfg := defaultConfig()
	cfg.Advisory.ModeOscillationLimit = 3
	cfg.Advisory.ModeOscillationWindowSec = 600
	obs := newObserver(cfg, testLogger())

	pressure := PressureSnapshot{Valid: true}
	for _, mode := range []string{"local_full", "cloud_first", "sequential", "cloud_first"} {
		_ = obs.computeAdvisories("cloud_first", mode, "idle", pressure)
		time.Sleep(2 * time.Millisecond)
	}

	advisories := obs.computeAdvisories("cloud_first", "sequential", "idle", pressure)
	found := false
	for _, adv := range advisories {
		if adv.Code == "mode_oscillation_risk" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected mode_oscillation_risk advisory")
	}
}

func TestFetchStatusViaUnix(t *testing.T) {
	tmpDir := t.TempDir()
	sock := filepath.Join(tmpDir, "observer-test.sock")

	ln, err := net.Listen("unix", sock)
	if err != nil {
		t.Fatalf("listen unix: %v", err)
	}
	defer ln.Close()

	done := make(chan struct{})
	go func() {
		defer close(done)
		conn, acceptErr := ln.Accept()
		if acceptErr != nil {
			return
		}
		defer conn.Close()

		_, _ = bufio.NewReader(conn).ReadBytes('\n')
		resp := map[string]any{
			"success": true,
			"result": map[string]any{
				"startup_modes": map[string]any{
					"desired_mode":   "cloud_only",
					"effective_mode": "sequential",
				},
			},
		}
		buf, _ := json.Marshal(resp)
		_, _ = conn.Write(append(buf, '\n'))
	}()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	status, err := fetchStatusViaUnix(ctx, sock)
	if err != nil {
		t.Fatalf("fetch status: %v", err)
	}
	modes, ok := status["startup_modes"].(map[string]any)
	if !ok {
		t.Fatalf("missing startup_modes")
	}
	if modes["desired_mode"] != "cloud_only" {
		t.Fatalf("unexpected desired mode: %v", modes["desired_mode"])
	}

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatalf("server goroutine did not finish")
	}

	_ = os.Remove(sock)
}
