package main

import (
	"context"
	"io"
	"log/slog"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func testLogger() *slog.Logger {
	return slog.New(slog.NewJSONHandler(io.Discard, nil))
}

func TestMemoryPressureBlocksWorkerStart(t *testing.T) {
	cfg := defaultConfig()
	cfg.Worker.Command = []string{"/bin/sh", "-c", "sleep 5"}
	cfg.Worker.AutoRecover = false
	cfg.Pressure.FailClosed = true
	cfg.Pressure.MinAvailableMemoryMB = 4096

	s := newSupervisor(cfg, testLogger())
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = s.stop(ctx)
	})

	s.samplePressureFn = func(context.Context) (PressureSnapshot, error) {
		return PressureSnapshot{
			Timestamp:         time.Now().UTC(),
			CPUPercent:        10,
			MemoryPercent:     95,
			MemoryAvailableMB: 256,
			MemoryTotalMB:     16384,
			Valid:             true,
		}, nil
	}

	_ = s.refreshPressure(context.Background())

	ctx, cancel := context.WithTimeout(context.Background(), 1500*time.Millisecond)
	defer cancel()
	err := s.startWorker(ctx, "test_low_memory")
	if err == nil {
		t.Fatalf("expected worker start to be blocked under pressure")
	}
	if !strings.Contains(err.Error(), "blocked by pressure gate") {
		t.Fatalf("expected pressure gate error, got: %v", err)
	}

	_, _, allowed, _ := s.snapshotForResponse()
	if allowed {
		t.Fatalf("expected heavy-load gate to remain closed")
	}
}

func TestCrashRecoveryWithBackoff(t *testing.T) {
	cfg := defaultConfig()
	cfg.Worker.Command = []string{"/bin/sh", "-c", "exit 1"}
	cfg.Worker.AutoRecover = true
	cfg.Worker.HealthURL = ""
	cfg.Backoff.InitialMs = 20
	cfg.Backoff.MaxMs = 60
	cfg.Backoff.Multiplier = 2.0
	cfg.Pressure.FailClosed = true

	s := newSupervisor(cfg, testLogger())
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = s.stop(ctx)
	})

	s.samplePressureFn = func(context.Context) (PressureSnapshot, error) {
		return PressureSnapshot{
			Timestamp:         time.Now().UTC(),
			CPUPercent:        5,
			MemoryPercent:     40,
			MemoryAvailableMB: 8192,
			MemoryTotalMB:     16384,
			Valid:             true,
		}, nil
	}

	_ = s.refreshPressure(context.Background())

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := s.startWorker(ctx, "test_crash"); err != nil {
		t.Fatalf("initial worker start failed: %v", err)
	}

	deadline := time.Now().Add(900 * time.Millisecond)
	for time.Now().Before(deadline) {
		starts := atomic.LoadUint64(&s.metrics.StartRequests)
		recoveries := atomic.LoadUint64(&s.metrics.CrashRecoveries)
		crashes := atomic.LoadUint64(&s.metrics.CrashCount)
		if starts >= 2 && recoveries >= 1 && crashes >= 1 {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}

	starts := atomic.LoadUint64(&s.metrics.StartRequests)
	recoveries := atomic.LoadUint64(&s.metrics.CrashRecoveries)
	crashes := atomic.LoadUint64(&s.metrics.CrashCount)
	t.Fatalf("expected crash recovery activity, got starts=%d recoveries=%d crashes=%d", starts, recoveries, crashes)
}
