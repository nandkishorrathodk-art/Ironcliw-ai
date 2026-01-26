"""
JARVIS Embedding Client for Cross-Repo Usage
==============================================

This module allows jarvis-prime and reactor-core to use JARVIS's centralized
embedding service instead of loading their own SentenceTransformer instances.

PROBLEM:
    Multiple SentenceTransformer instances across repos cause:
    1. Memory waste (each instance is ~500MB-1GB)
    2. Semaphore leaks from internal multiprocessing pools
    3. Potential OOM kills on memory-constrained systems

SOLUTION:
    Use JARVIS's centralized EmbeddingService via:
    1. Unix socket (fastest, recommended)
    2. HTTP API (fallback when socket unavailable)
    3. Direct Python import (when running in same process)

Usage in jarvis-prime/reactor-core:
    # Copy this file to your repo's appropriate location
    from trinity_clients.jarvis_embedding_client import get_embeddings
    
    embeddings = await get_embeddings(["text1", "text2"])

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import socket
import struct
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Unix socket path for local communication
EMBEDDING_SOCKET = "/tmp/jarvis_embedding_service.sock"

# HTTP fallback endpoint
EMBEDDING_HTTP_URL = os.getenv("JARVIS_EMBEDDING_URL", "http://127.0.0.1:8010/api/embeddings")

# Timeout for socket operations
SOCKET_TIMEOUT = float(os.getenv("JARVIS_EMBEDDING_TIMEOUT", "30.0"))

# Whether to fall back to local model if JARVIS is unavailable
ALLOW_LOCAL_FALLBACK = os.getenv("JARVIS_EMBEDDING_LOCAL_FALLBACK", "false").lower() == "true"


# =============================================================================
# UNIX SOCKET CLIENT
# =============================================================================

class UnixSocketEmbeddingClient:
    """
    Client for JARVIS embedding service via Unix socket.
    
    This is the fastest method for cross-repo embedding access.
    """
    
    def __init__(self, socket_path: str = EMBEDDING_SOCKET):
        self.socket_path = socket_path
        self._connected = False
    
    def is_available(self) -> bool:
        """Check if the embedding service socket is available."""
        return Path(self.socket_path).exists()
    
    async def encode(self, texts: Union[str, List[str]]) -> Optional[np.ndarray]:
        """
        Encode texts using JARVIS embedding service via Unix socket.
        
        Args:
            texts: Text or list of texts to encode
            
        Returns:
            Numpy array of embeddings or None if service unavailable
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not self.is_available():
            logger.debug("[EmbeddingClient] Socket not available")
            return None
        
        try:
            # Connect to socket
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self.socket_path),
                timeout=5.0
            )
            
            # Send request
            request = {"action": "encode", "texts": texts}
            request_data = json.dumps(request).encode('utf-8')
            
            # Send length-prefixed message
            writer.write(struct.pack('>I', len(request_data)))
            writer.write(request_data)
            await writer.drain()
            
            # Receive response
            length_data = await asyncio.wait_for(
                reader.readexactly(4),
                timeout=SOCKET_TIMEOUT
            )
            response_length = struct.unpack('>I', length_data)[0]
            
            response_data = await asyncio.wait_for(
                reader.readexactly(response_length),
                timeout=SOCKET_TIMEOUT
            )
            
            # Close connection
            writer.close()
            await writer.wait_closed()
            
            # Parse response (embeddings are pickled numpy arrays)
            response = pickle.loads(response_data)
            
            if response.get("success"):
                return np.array(response["embeddings"])
            else:
                logger.error(f"[EmbeddingClient] Server error: {response.get('error')}")
                return None
                
        except asyncio.TimeoutError:
            logger.warning("[EmbeddingClient] Socket timeout")
            return None
        except Exception as e:
            logger.warning(f"[EmbeddingClient] Socket error: {e}")
            return None


# =============================================================================
# HTTP CLIENT (Fallback)
# =============================================================================

class HttpEmbeddingClient:
    """
    Fallback HTTP client for JARVIS embedding service.
    
    Used when Unix socket is not available.
    """
    
    def __init__(self, url: str = EMBEDDING_HTTP_URL):
        self.url = url
        self._session = None
    
    async def encode(self, texts: Union[str, List[str]]) -> Optional[np.ndarray]:
        """Encode texts using HTTP API."""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url,
                    json={"texts": texts},
                    timeout=aiohttp.ClientTimeout(total=SOCKET_TIMEOUT)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return np.array(data["embeddings"])
                    else:
                        logger.error(f"[EmbeddingClient] HTTP error: {response.status}")
                        return None
                        
        except ImportError:
            logger.warning("[EmbeddingClient] aiohttp not available for HTTP fallback")
            return None
        except Exception as e:
            logger.warning(f"[EmbeddingClient] HTTP error: {e}")
            return None


# =============================================================================
# UNIFIED CLIENT
# =============================================================================

class JARVISEmbeddingClient:
    """
    Unified embedding client that tries multiple transport methods.
    
    Priority:
    1. Direct import (if running in JARVIS process)
    2. Unix socket (fastest cross-process)
    3. HTTP API (fallback)
    4. Local SentenceTransformer (if allowed and all else fails)
    """
    
    def __init__(self):
        self._socket_client = UnixSocketEmbeddingClient()
        self._http_client = HttpEmbeddingClient()
        self._local_model = None
        self._tried_direct_import = False
        self._direct_service = None
    
    async def _try_direct_import(self):
        """Try to import JARVIS embedding service directly."""
        if self._tried_direct_import:
            return self._direct_service
        
        self._tried_direct_import = True
        
        try:
            from backend.core.embedding_service import get_embedding_service
            self._direct_service = await get_embedding_service()
            logger.info("[EmbeddingClient] Using direct import (same process)")
        except ImportError:
            pass
        
        return self._direct_service
    
    async def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Encode texts using the best available method.
        
        Args:
            texts: Text or list of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings or None if all methods fail
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Method 1: Direct import
        service = await self._try_direct_import()
        if service:
            result = await service.encode(texts, batch_size=batch_size)
            if result is not None:
                return result
        
        # Method 2: Unix socket
        if self._socket_client.is_available():
            result = await self._socket_client.encode(texts)
            if result is not None:
                return result
        
        # Method 3: HTTP API
        result = await self._http_client.encode(texts)
        if result is not None:
            return result
        
        # Method 4: Local fallback (if allowed)
        if ALLOW_LOCAL_FALLBACK:
            return self._encode_local(texts)
        
        logger.error("[EmbeddingClient] All embedding methods failed")
        return None
    
    def _encode_local(self, texts: List[str]) -> Optional[np.ndarray]:
        """Fallback to local SentenceTransformer (creates memory pressure)."""
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.warning(
                    "[EmbeddingClient] Falling back to local SentenceTransformer - "
                    "this may cause memory issues"
                )
                self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.error("[EmbeddingClient] sentence-transformers not available")
                return None
        
        try:
            return self._local_model.encode(texts, show_progress_bar=False)
        except Exception as e:
            logger.error(f"[EmbeddingClient] Local encoding failed: {e}")
            return None
    
    def cleanup(self):
        """Clean up local resources if any."""
        if self._local_model is not None:
            self._local_model = None
            import gc
            gc.collect()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_client: Optional[JARVISEmbeddingClient] = None


def get_embedding_client() -> JARVISEmbeddingClient:
    """Get the global embedding client singleton."""
    global _client
    if _client is None:
        _client = JARVISEmbeddingClient()
    return _client


async def get_embeddings(
    texts: Union[str, List[str]],
    batch_size: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Convenience function to get embeddings.
    
    Usage:
        embeddings = await get_embeddings(["text1", "text2"])
    """
    client = get_embedding_client()
    return await client.encode(texts, batch_size=batch_size)


def cleanup_embedding_client():
    """Clean up the embedding client."""
    global _client
    if _client is not None:
        _client.cleanup()
        _client = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "JARVISEmbeddingClient",
    "UnixSocketEmbeddingClient",
    "HttpEmbeddingClient",
    "get_embedding_client",
    "get_embeddings",
    "cleanup_embedding_client",
]
