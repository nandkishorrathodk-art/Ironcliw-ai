#!/usr/bin/env python3
"""
Cloud ECAPA Speaker Embedding Service
======================================

Robust, production-ready cloud service for ECAPA-TDNN speaker embeddings.
Designed to run on GCP Cloud Run or any container platform.

Features:
- Async FastAPI server with health checks
- ECAPA-TDNN model with lazy loading and caching
- Circuit breaker for downstream failures
- Request batching for efficiency
- Embedding caching with TTL
- Comprehensive metrics and telemetry
- Dynamic configuration via environment

v18.0.0 - Full Production Release

Endpoints:
    GET  /health              - Health check with ECAPA readiness
    GET  /status              - Detailed service status
    POST /api/ml/speaker_embedding  - Extract speaker embedding
    POST /api/ml/speaker_verify     - Verify speaker against reference
    POST /api/ml/batch_embedding    - Batch embedding extraction

Environment Variables:
    ECAPA_MODEL_PATH      - Path to ECAPA model (default: speechbrain/spkrec-ecapa-voxceleb)
    ECAPA_CACHE_DIR       - Cache directory for model files
    ECAPA_DEVICE          - Device to run on (cpu/cuda/mps)
    ECAPA_BATCH_SIZE      - Max batch size for processing
    ECAPA_CACHE_TTL       - Embedding cache TTL in seconds
    ECAPA_WARMUP_ON_START - Run warmup inference on startup
    PORT                  - Server port (default: 8010)
"""

import asyncio
import base64
import hashlib
import logging
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import traceback

import numpy as np

# Pre-import heavy ML libraries at module load time to avoid thread pool issues
# This happens once when the container starts, before any requests
print(">>> Pre-importing torch at module level...", flush=True)
import torch
print(f">>> torch {torch.__version__} imported successfully", flush=True)

print(">>> Pre-importing speechbrain at module level...", flush=True)
from speechbrain.inference.speaker import EncoderClassifier
print(">>> speechbrain imported successfully", flush=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ecapa_cloud_service")

# =============================================================================
# DYNAMIC CONFIGURATION
# =============================================================================

class CloudECAPAConfig:
    """
    Dynamic configuration from environment variables.

    Supports robust pre-baked model cache detection with automatic fallback.
    Priority order for cache locations:
    1. ECAPA_SOURCE_CACHE (Docker pre-baked: /opt/ecapa_cache)
    2. ECAPA_CACHE_DIR (Runtime: /tmp/ecapa_cache)
    3. Home directory fallback (~/.cache/ecapa)
    4. Fresh download from HuggingFace (last resort)
    """

    MODEL_PATH = os.getenv("ECAPA_MODEL_PATH", "speechbrain/spkrec-ecapa-voxceleb")

    # Cache locations (priority order)
    SOURCE_CACHE = os.getenv("ECAPA_SOURCE_CACHE", "/opt/ecapa_cache")  # Docker pre-baked
    CACHE_DIR = os.getenv("ECAPA_CACHE_DIR", "/tmp/ecapa_cache")  # Runtime cache
    HOME_CACHE = os.path.expanduser("~/.cache/ecapa")  # Fallback

    DEVICE = os.getenv("ECAPA_DEVICE", "cpu")  # cpu, cuda, mps
    BATCH_SIZE = int(os.getenv("ECAPA_BATCH_SIZE", "8"))
    CACHE_TTL = int(os.getenv("ECAPA_CACHE_TTL", "3600"))  # 1 hour
    CACHE_MAX_SIZE = int(os.getenv("ECAPA_CACHE_MAX_SIZE", "1000"))
    WARMUP_ON_START = os.getenv("ECAPA_WARMUP_ON_START", "true").lower() == "true"
    PORT = int(os.getenv("PORT", "8010"))

    # Circuit breaker settings
    CB_FAILURE_THRESHOLD = int(os.getenv("ECAPA_CB_FAILURES", "5"))
    CB_RECOVERY_TIMEOUT = float(os.getenv("ECAPA_CB_RECOVERY", "30.0"))

    # Request settings
    REQUEST_TIMEOUT = float(os.getenv("ECAPA_REQUEST_TIMEOUT", "30.0"))
    SAMPLE_RATE = int(os.getenv("ECAPA_SAMPLE_RATE", "16000"))

    # Pre-baked cache settings
    SKIP_HF_DOWNLOAD = os.getenv("ECAPA_SKIP_HF_DOWNLOAD", "true").lower() == "true"
    CACHE_VERIFICATION_ENABLED = os.getenv("ECAPA_VERIFY_CACHE", "true").lower() == "true"

    # Required files for a valid pre-baked cache
    REQUIRED_CACHE_FILES = [
        "hyperparams.yaml",
        "embedding_model.ckpt",
    ]

    # Optional but expected files
    OPTIONAL_CACHE_FILES = [
        "mean_var_norm_emb.ckpt",
        "classifier.ckpt",
        "label_encoder.txt",
    ]

    @classmethod
    def get_cache_locations(cls) -> List[str]:
        """Get all possible cache locations in priority order."""
        locations = []

        # Priority 1: Docker pre-baked source cache
        if cls.SOURCE_CACHE:
            locations.append(cls.SOURCE_CACHE)

        # Priority 2: Runtime cache (may have been populated by entrypoint.sh)
        if cls.CACHE_DIR and cls.CACHE_DIR != cls.SOURCE_CACHE:
            locations.append(cls.CACHE_DIR)

        # Priority 3: Home directory fallback
        if cls.HOME_CACHE not in locations:
            locations.append(cls.HOME_CACHE)

        return locations

    @classmethod
    def verify_cache_integrity(cls, cache_path: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Verify that a cache directory contains all required model files.

        Returns:
            Tuple of (is_valid, message, diagnostics)
        """
        diagnostics = {
            "path": cache_path,
            "exists": os.path.isdir(cache_path),
            "required_files": {},
            "optional_files": {},
            "total_size_mb": 0,
        }

        if not os.path.isdir(cache_path):
            return False, f"Cache directory does not exist: {cache_path}", diagnostics

        # Check required files
        missing_required = []
        for filename in cls.REQUIRED_CACHE_FILES:
            filepath = os.path.join(cache_path, filename)
            exists = os.path.isfile(filepath)
            # Get file size safely (handle permission errors)
            file_size_kb = 0
            if exists:
                try:
                    file_size_kb = round(os.path.getsize(filepath) / 1024, 2)
                except (OSError, PermissionError) as e:
                    logger.debug(f"Could not get size of {filepath}: {e}")
                    file_size_kb = -1  # Indicate error but file exists

            diagnostics["required_files"][filename] = {
                "exists": exists,
                "size_kb": file_size_kb,
            }
            if not exists:
                missing_required.append(filename)

        # Check optional files
        for filename in cls.OPTIONAL_CACHE_FILES:
            filepath = os.path.join(cache_path, filename)
            exists = os.path.isfile(filepath)
            file_size_kb = 0
            if exists:
                try:
                    file_size_kb = round(os.path.getsize(filepath) / 1024, 2)
                except (OSError, PermissionError) as e:
                    logger.debug(f"Could not get size of {filepath}: {e}")
                    file_size_kb = -1
            diagnostics["optional_files"][filename] = {
                "exists": exists,
                "size_kb": file_size_kb,
            }

        # Calculate total size (with permission error handling)
        total_size = 0
        for root, dirs, files in os.walk(cache_path):
            for f in files:
                try:
                    total_size += os.path.getsize(os.path.join(root, f))
                except (OSError, PermissionError):
                    pass  # Skip files we can't read
        diagnostics["total_size_mb"] = round(total_size / (1024 * 1024), 2)

        if missing_required:
            return False, f"Missing required files: {missing_required}", diagnostics

        return True, "Cache verified successfully", diagnostics

    @classmethod
    def find_valid_cache(cls) -> Tuple[Optional[str], str, Dict[str, Any]]:
        """
        Find the first valid cache location with all required files.

        Returns:
            Tuple of (cache_path or None, message, diagnostics)
        """
        all_diagnostics = {}

        for cache_path in cls.get_cache_locations():
            is_valid, message, diag = cls.verify_cache_integrity(cache_path)
            all_diagnostics[cache_path] = diag

            if is_valid:
                logger.info(f"✅ Found valid pre-baked cache: {cache_path}")
                logger.info(f"   Size: {diag['total_size_mb']}MB")
                return cache_path, message, all_diagnostics
            else:
                logger.debug(f"Cache not valid at {cache_path}: {message}")

        return None, "No valid pre-baked cache found", all_diagnostics

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        valid_cache, cache_msg, _ = cls.find_valid_cache()
        return {
            "model_path": cls.MODEL_PATH,
            "cache_dir": cls.CACHE_DIR,
            "source_cache": cls.SOURCE_CACHE,
            "valid_cache_found": valid_cache,
            "cache_status": cache_msg,
            "device": cls.DEVICE,
            "batch_size": cls.BATCH_SIZE,
            "cache_ttl": cls.CACHE_TTL,
            "warmup_on_start": cls.WARMUP_ON_START,
            "skip_hf_download": cls.SKIP_HF_DOWNLOAD,
            "port": cls.PORT,
        }


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, reject requests
    HALF_OPEN = auto()   # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for ECAPA model operations."""

    failure_threshold: int = CloudECAPAConfig.CB_FAILURE_THRESHOLD
    recovery_timeout: float = CloudECAPAConfig.CB_RECOVERY_TIMEOUT

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    success_count: int = 0

    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.success_count += 1
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker: CLOSED (recovered)")

    def record_failure(self, error: str = None):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker: OPEN (failures={self.failure_count}, error={error})")

    def can_execute(self) -> bool:
        """Check if operation can proceed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker: HALF_OPEN (testing recovery)")
                    return True
            return False

        # HALF_OPEN: allow one request to test
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.name,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time,
        }


# =============================================================================
# LRU CACHE WITH TTL
# =============================================================================

class TTLCache:
    """LRU cache with time-to-live for embeddings."""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self._lock = asyncio.Lock()

        # Stats
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache if exists and not expired."""
        async with self._lock:
            if key not in self.cache:
                self.misses += 1
                return None

            # Check expiry
            if time.time() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                self.misses += 1
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

    async def set(self, key: str, value: Any):
        """Set item in cache with TTL."""
        async with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    # Remove oldest item
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]

            self.cache[key] = value
            self.timestamps[key] = time.time()

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate * 100:.1f}%",
        }


# =============================================================================
# ECAPA MODEL MANAGER
# =============================================================================

class ECAPAModelManager:
    """
    Manages ECAPA-TDNN model lifecycle with lazy loading and caching.

    Features:
    - Automatic pre-baked cache detection (Docker-optimized)
    - Multi-location cache fallback
    - HuggingFace download bypass when cache is valid
    - Thread-safe async loading
    - Comprehensive telemetry
    """

    def __init__(self):
        self.model = None
        self.device = CloudECAPAConfig.DEVICE
        self.model_path = CloudECAPAConfig.MODEL_PATH

        # Cache discovery - find the best available cache
        self._prebaked_cache: Optional[str] = None
        self._cache_diagnostics: Dict[str, Any] = {}
        self._using_prebaked = False

        # Will be set during initialization after cache discovery
        self.cache_dir = CloudECAPAConfig.CACHE_DIR

        self._loading = False
        self._load_lock = asyncio.Lock()
        self._ready = False
        self._error: Optional[str] = None

        # Telemetry
        self.load_time_ms: Optional[float] = None
        self.warmup_time_ms: Optional[float] = None
        self.inference_count = 0
        self.total_inference_time_ms = 0.0
        self.load_source: str = "unknown"  # "prebaked", "runtime_cache", "fresh_download"

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker()

        # Embedding cache
        self.embedding_cache = TTLCache(
            max_size=CloudECAPAConfig.CACHE_MAX_SIZE,
            ttl=CloudECAPAConfig.CACHE_TTL
        )

    @property
    def is_ready(self) -> bool:
        return self._ready and self.model is not None

    @property
    def avg_inference_ms(self) -> float:
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_time_ms / self.inference_count

    async def initialize(self) -> bool:
        """Initialize and load the ECAPA model."""
        async with self._load_lock:
            if self._ready:
                return True

            if self._loading:
                # Wait for ongoing load
                for _ in range(60):  # 60 second timeout
                    await asyncio.sleep(1)
                    if self._ready:
                        return True
                return False

            self._loading = True

            try:
                logger.info("=" * 60)
                logger.info("ECAPA-TDNN Cloud Service Initialization")
                logger.info("=" * 60)
                logger.info(f"Model: {self.model_path}")
                logger.info(f"Device: {self.device}")
                logger.info(f"Cache: {self.cache_dir}")

                start_time = time.time()

                # Ensure cache directory exists
                os.makedirs(self.cache_dir, exist_ok=True)

                # Load SpeechBrain ECAPA model
                await self._load_model()

                self.load_time_ms = (time.time() - start_time) * 1000
                logger.info(f"Model loaded in {self.load_time_ms:.0f}ms")

                # Run warmup if enabled
                if CloudECAPAConfig.WARMUP_ON_START:
                    await self._warmup()

                self._ready = True
                logger.info("=" * 60)
                logger.info("ECAPA-TDNN Ready for Inference")
                logger.info("=" * 60)

                return True

            except Exception as e:
                self._error = str(e)
                logger.error(f"ECAPA initialization failed: {e}")
                logger.error(traceback.format_exc())
                return False
            finally:
                self._loading = False

    async def _load_model(self):
        """Load the ECAPA-TDNN model (runs in thread pool)."""
        logger.info("Starting model load in thread pool...")
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_model_sync)
            logger.info("Thread pool model load completed")
        except Exception as e:
            logger.error(f"Thread pool model load failed: {e}")
            raise

    def _load_model_sync(self):
        """
        Synchronous model loading with pre-baked cache detection.

        Priority:
        1. Use pre-baked Docker cache if available (fastest - no download)
        2. Use runtime cache if populated by entrypoint.sh
        3. Fall back to fresh HuggingFace download (slowest)
        """
        logger.info("_load_model_sync: Function entered")
        logger.info("=" * 50)
        logger.info("ECAPA MODEL LOADING - Cache Detection")
        logger.info("=" * 50)

        # Step 1: Detect best available cache
        prebaked_cache, cache_msg, all_diag = CloudECAPAConfig.find_valid_cache()
        self._cache_diagnostics = all_diag

        if prebaked_cache:
            logger.info(f"✅ PRE-BAKED CACHE DETECTED: {prebaked_cache}")
            self._prebaked_cache = prebaked_cache
            self._using_prebaked = True
            self.cache_dir = prebaked_cache
            self.load_source = "prebaked"

            # Set HuggingFace offline mode to prevent any downloads
            if CloudECAPAConfig.SKIP_HF_DOWNLOAD:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                logger.info("   HuggingFace download disabled (using pre-baked cache)")
        else:
            logger.warning(f"⚠️ No pre-baked cache found: {cache_msg}")
            logger.info("   Will attempt fresh download from HuggingFace")
            self.load_source = "fresh_download"

            # Ensure runtime cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)

        # Step 2: Determine device
        if self.device == "mps" and torch.backends.mps.is_available():
            device = "mps"
        elif self.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        logger.info(f"   Device: {device}")
        logger.info(f"   Cache dir: {self.cache_dir}")
        logger.info(f"   Model source: {self.model_path}")
        logger.info("=" * 50)

        # Step 3: Load model
        load_start = time.time()

        try:
            logger.info("Loading ECAPA-TDNN model...")

            self.model = EncoderClassifier.from_hparams(
                source=self.model_path,
                savedir=self.cache_dir,
                run_opts={"device": device}
            )

            load_duration = (time.time() - load_start) * 1000
            self.device = device

            logger.info("=" * 50)
            logger.info(f"✅ ECAPA-TDNN LOADED SUCCESSFULLY")
            logger.info(f"   Device: {device}")
            logger.info(f"   Source: {self.load_source}")
            logger.info(f"   Load time: {load_duration:.0f}ms")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")

            # If pre-baked cache failed, try fresh download as fallback
            if self._using_prebaked:
                logger.warning("🔄 Pre-baked cache load failed, attempting fresh download...")

                # Disable offline mode
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)

                # Use runtime cache for fresh download
                self.cache_dir = CloudECAPAConfig.CACHE_DIR
                os.makedirs(self.cache_dir, exist_ok=True)
                self.load_source = "fresh_download_fallback"

                self.model = EncoderClassifier.from_hparams(
                    source=self.model_path,
                    savedir=self.cache_dir,
                    run_opts={"device": device}
                )

                self.device = device
                logger.info(f"✅ ECAPA-TDNN loaded via fresh download fallback on {device}")
            else:
                raise

    async def _warmup(self):
        """Run warmup inference to ensure model is fully loaded."""
        logger.info("Running warmup inference...")

        start_time = time.time()

        # Generate test audio (1 second of silence)
        test_audio = np.zeros(16000, dtype=np.float32)

        try:
            # Run warmup synchronously to avoid thread pool executor issues during startup
            logger.info("Warmup: Converting audio to tensor...")
            audio_tensor = torch.tensor(test_audio).unsqueeze(0)

            logger.info("Warmup: Running encode_batch (this may take 30-60s on first run)...")
            with torch.no_grad():
                embedding = self.model.encode_batch(audio_tensor).squeeze().cpu().numpy()

            logger.info(f"Warmup: Got embedding shape {embedding.shape}")
            self.warmup_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Warmup completed successfully in {self.warmup_time_ms:.0f}ms")
        except Exception as e:
            logger.warning(f"Warmup inference failed (non-critical): {e}")
            import traceback
            logger.warning(traceback.format_exc())

    async def extract_embedding(
        self,
        audio_data: np.ndarray,
        use_cache: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio.

        Args:
            audio_data: Audio as numpy array (float32, 16kHz)
            use_cache: Whether to check/use embedding cache

        Returns:
            192-dimensional speaker embedding or None on failure
        """
        if not self.is_ready:
            if not await self.initialize():
                raise RuntimeError("ECAPA model not available")

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise RuntimeError(f"Circuit breaker OPEN: {self.circuit_breaker.failure_count} failures")

        # Check cache
        if use_cache:
            cache_key = self._compute_audio_hash(audio_data)
            cached = await self.embedding_cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for audio hash {cache_key[:8]}...")
                return cached

        # Extract embedding
        start_time = time.time()

        try:
            import torch

            # Ensure audio is normalized float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize if needed
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val

            # Convert to tensor
            audio_tensor = torch.tensor(audio_data).unsqueeze(0)

            # Run in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self.model.encode_batch(audio_tensor).squeeze().cpu().numpy()
            )

            # Update stats
            inference_time = (time.time() - start_time) * 1000
            self.inference_count += 1
            self.total_inference_time_ms += inference_time

            self.circuit_breaker.record_success()

            # Cache result
            if use_cache:
                await self.embedding_cache.set(cache_key, embedding)

            logger.debug(f"Embedding extracted in {inference_time:.0f}ms, shape: {embedding.shape}")

            return embedding

        except Exception as e:
            self.circuit_breaker.record_failure(str(e))
            logger.error(f"Embedding extraction failed: {e}")
            raise

    def _compute_audio_hash(self, audio: np.ndarray) -> str:
        """Compute hash of audio for caching."""
        return hashlib.sha256(audio.tobytes()).hexdigest()

    async def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def status(self) -> Dict[str, Any]:
        """Get detailed model status including cache diagnostics."""
        return {
            "ready": self.is_ready,
            "loading": self._loading,
            "error": self._error,
            "device": self.device,
            "model_path": self.model_path,
            "load_time_ms": self.load_time_ms,
            "warmup_time_ms": self.warmup_time_ms,
            "inference_count": self.inference_count,
            "avg_inference_ms": round(self.avg_inference_ms, 2),
            "circuit_breaker": self.circuit_breaker.to_dict(),
            "embedding_cache": self.embedding_cache.stats(),
            # Pre-baked cache information
            "model_cache": {
                "cache_dir": self.cache_dir,
                "load_source": self.load_source,
                "using_prebaked": self._using_prebaked,
                "prebaked_cache_path": self._prebaked_cache,
                "diagnostics": self._cache_diagnostics,
            },
        }


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Global model manager
_model_manager: Optional[ECAPAModelManager] = None


def get_model_manager() -> ECAPAModelManager:
    """Get or create the global model manager."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ECAPAModelManager()
    return _model_manager


# Create FastAPI app
try:
    from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available - HTTP server disabled")


if FASTAPI_AVAILABLE:

    app = FastAPI(
        title="ECAPA Cloud Service",
        description="Cloud ECAPA-TDNN Speaker Embedding Service",
        version="18.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # =========================================================================
    # Request/Response Models
    # =========================================================================

    class EmbeddingRequest(BaseModel):
        """Request for speaker embedding extraction."""
        audio_data: str = Field(..., description="Base64 encoded audio bytes")
        sample_rate: int = Field(default=16000, description="Audio sample rate")
        format: str = Field(default="float32", description="Audio format (int16, float32)")
        test_mode: bool = Field(default=False, description="Test mode flag")

    class EmbeddingResponse(BaseModel):
        """Response with extracted embedding."""
        success: bool
        embedding: Optional[List[float]] = None
        embedding_size: Optional[int] = None
        processing_time_ms: Optional[float] = None
        cached: bool = False
        error: Optional[str] = None

    class VerifyRequest(BaseModel):
        """Request for speaker verification."""
        audio_data: str = Field(..., description="Base64 encoded audio to verify")
        reference_embedding: List[float] = Field(..., description="Reference embedding")
        sample_rate: int = Field(default=16000)
        format: str = Field(default="float32")

    class VerifyResponse(BaseModel):
        """Response with verification result."""
        success: bool
        verified: bool = False
        similarity: float = 0.0
        confidence: float = 0.0
        threshold: float = 0.85
        processing_time_ms: Optional[float] = None
        error: Optional[str] = None

    class BatchEmbeddingRequest(BaseModel):
        """Request for batch embedding extraction."""
        audio_samples: List[str] = Field(..., description="List of base64 encoded audio")
        sample_rate: int = Field(default=16000)
        format: str = Field(default="float32")

    class BatchEmbeddingResponse(BaseModel):
        """Response with batch embeddings."""
        success: bool
        embeddings: List[Optional[List[float]]] = []
        processing_time_ms: Optional[float] = None
        error: Optional[str] = None

    # =========================================================================
    # Health & Status Endpoints
    # =========================================================================

    @app.on_event("startup")
    async def startup_event():
        """Initialize model on startup."""
        logger.info("Starting ECAPA Cloud Service...")
        manager = get_model_manager()

        # Start initialization in background with exception handling
        async def init_with_logging():
            try:
                logger.info("Background initialization task starting...")
                result = await manager.initialize()
                logger.info(f"Background initialization completed: {result}")
            except Exception as e:
                logger.error(f"Background initialization failed: {e}")
                import traceback
                logger.error(traceback.format_exc())

        asyncio.create_task(init_with_logging())

    @app.get("/health")
    async def health_check():
        """Health check endpoint with cache status."""
        manager = get_model_manager()

        response = {
            "status": "healthy" if manager.is_ready else "initializing",
            "ecapa_ready": manager.is_ready,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Include cache info if model is loaded
        if manager.is_ready:
            response["load_source"] = manager.load_source
            response["using_prebaked_cache"] = manager._using_prebaked
            response["load_time_ms"] = manager.load_time_ms

        return response

    @app.get("/status")
    async def get_status():
        """Detailed service status."""
        manager = get_model_manager()

        return {
            "service": "ecapa_cloud_service",
            "version": "18.0.0",
            "config": CloudECAPAConfig.to_dict(),
            "model": manager.status(),
            "timestamp": datetime.utcnow().isoformat(),
        }

    @app.get("/api/ml/health")
    async def ml_health():
        """ML API health endpoint (for compatibility)."""
        manager = get_model_manager()

        return {
            "status": "healthy" if manager.is_ready else "initializing",
            "ecapa_ready": manager.is_ready,
            "circuit_breaker": manager.circuit_breaker.state.name,
        }

    # =========================================================================
    # Embedding Endpoints
    # =========================================================================

    @app.post("/api/ml/speaker_embedding", response_model=EmbeddingResponse)
    async def extract_embedding(request: EmbeddingRequest):
        """Extract speaker embedding from audio."""
        start_time = time.time()

        manager = get_model_manager()

        try:
            # Ensure model is ready
            if not manager.is_ready:
                if not await manager.initialize():
                    raise HTTPException(
                        status_code=503,
                        detail="ECAPA model not available"
                    )

            # Decode audio
            try:
                audio_bytes = base64.b64decode(request.audio_data)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid base64 audio data: {e}"
                )

            # Convert to numpy array
            if request.format == "int16":
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio = np.frombuffer(audio_bytes, dtype=np.float32)

            # Resample if needed (simple linear interpolation)
            if request.sample_rate != 16000:
                duration = len(audio) / request.sample_rate
                new_length = int(duration * 16000)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, new_length),
                    np.arange(len(audio)),
                    audio
                )

            # Extract embedding
            embedding = await manager.extract_embedding(audio)

            if embedding is None:
                raise HTTPException(
                    status_code=500,
                    detail="Embedding extraction returned None"
                )

            processing_time = (time.time() - start_time) * 1000

            return EmbeddingResponse(
                success=True,
                embedding=embedding.tolist(),
                embedding_size=len(embedding),
                processing_time_ms=round(processing_time, 2),
                cached=False,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            logger.error(traceback.format_exc())
            return EmbeddingResponse(
                success=False,
                error=str(e),
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
            )

    @app.post("/api/ml/speaker_verify", response_model=VerifyResponse)
    async def verify_speaker(request: VerifyRequest):
        """Verify speaker against reference embedding."""
        start_time = time.time()

        manager = get_model_manager()

        try:
            # Ensure model is ready
            if not manager.is_ready:
                if not await manager.initialize():
                    raise HTTPException(
                        status_code=503,
                        detail="ECAPA model not available"
                    )

            # Decode audio
            audio_bytes = base64.b64decode(request.audio_data)

            if request.format == "int16":
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio = np.frombuffer(audio_bytes, dtype=np.float32)

            # Resample if needed
            if request.sample_rate != 16000:
                duration = len(audio) / request.sample_rate
                new_length = int(duration * 16000)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, new_length),
                    np.arange(len(audio)),
                    audio
                )

            # Extract embedding
            embedding = await manager.extract_embedding(audio)

            if embedding is None:
                raise HTTPException(
                    status_code=500,
                    detail="Embedding extraction failed"
                )

            # Compare with reference
            reference = np.array(request.reference_embedding, dtype=np.float32)
            similarity = await manager.compute_similarity(embedding, reference)

            # Convert similarity to confidence (0-1 range)
            # Similarity is cosine similarity (-1 to 1), normalize to 0-1
            confidence = (similarity + 1) / 2

            threshold = 0.85
            verified = confidence >= threshold

            processing_time = (time.time() - start_time) * 1000

            return VerifyResponse(
                success=True,
                verified=verified,
                similarity=round(similarity, 4),
                confidence=round(confidence, 4),
                threshold=threshold,
                processing_time_ms=round(processing_time, 2),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Speaker verification error: {e}")
            return VerifyResponse(
                success=False,
                error=str(e),
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
            )

    @app.post("/api/ml/batch_embedding", response_model=BatchEmbeddingResponse)
    async def extract_batch_embeddings(request: BatchEmbeddingRequest):
        """Extract embeddings for multiple audio samples."""
        start_time = time.time()

        manager = get_model_manager()

        try:
            if not manager.is_ready:
                if not await manager.initialize():
                    raise HTTPException(
                        status_code=503,
                        detail="ECAPA model not available"
                    )

            embeddings = []

            for audio_b64 in request.audio_samples:
                try:
                    audio_bytes = base64.b64decode(audio_b64)

                    if request.format == "int16":
                        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    else:
                        audio = np.frombuffer(audio_bytes, dtype=np.float32)

                    if request.sample_rate != 16000:
                        duration = len(audio) / request.sample_rate
                        new_length = int(duration * 16000)
                        audio = np.interp(
                            np.linspace(0, len(audio) - 1, new_length),
                            np.arange(len(audio)),
                            audio
                        )

                    embedding = await manager.extract_embedding(audio)
                    embeddings.append(embedding.tolist() if embedding is not None else None)

                except Exception as e:
                    logger.warning(f"Batch item failed: {e}")
                    embeddings.append(None)

            processing_time = (time.time() - start_time) * 1000

            return BatchEmbeddingResponse(
                success=True,
                embeddings=embeddings,
                processing_time_ms=round(processing_time, 2),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Batch extraction error: {e}")
            return BatchEmbeddingResponse(
                success=False,
                error=str(e),
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
            )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run the ECAPA cloud service."""
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not available. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("ECAPA Cloud Service v18.0.0")
    logger.info("=" * 60)
    logger.info(f"Configuration: {CloudECAPAConfig.to_dict()}")

    uvicorn.run(
        "ecapa_cloud_service:app",
        host="0.0.0.0",
        port=CloudECAPAConfig.PORT,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
