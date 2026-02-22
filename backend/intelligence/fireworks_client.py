"""
Fireworks AI Client v1.0
=========================

Fast inference client for Fireworks AI platform - specialized for open-source models.

Fireworks AI provides:
    - Ultra-fast inference (~2x faster than standard APIs)
    - Open-source model hosting (Llama, Mixtral, Qwen, etc.)
    - Competitive pricing (50-80% cheaper than Claude/GPT)
    - Function calling support
    - Streaming responses

Integration into JARVIS:
    - Acts as fallback between J-Prime and Claude
    - Routing: PRIME_API → FIREWORKS → CLAUDE
    - Preferred for code generation and reasoning tasks
    - Cost-effective alternative for high-volume use

Author: JARVIS System (Windows Port)
Version: 1.0.0
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Configuration from environment
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_API_BASE = os.getenv("FIREWORKS_API_BASE", "https://api.fireworks.ai/inference/v1")
FIREWORKS_DEFAULT_MODEL = os.getenv("FIREWORKS_DEFAULT_MODEL", "accounts/fireworks/models/llama-v3p3-70b-instruct")
FIREWORKS_TIMEOUT_SECONDS = float(os.getenv("FIREWORKS_TIMEOUT_SECONDS", "60.0"))
FIREWORKS_MAX_TOKENS = int(os.getenv("FIREWORKS_MAX_TOKENS", "4096"))
FIREWORKS_ENABLED = os.getenv("FIREWORKS_ENABLED", "true").lower() == "true"


# Cost tracking (per 1M tokens) - Fireworks pricing as of 2026
FIREWORKS_COST_PER_1M_INPUT = {
    "accounts/fireworks/models/llama-v3p3-70b-instruct": 0.90,  # Llama 3.3 70B
    "accounts/fireworks/models/qwen2p5-72b-instruct": 0.90,     # Qwen 2.5 72B
    "accounts/fireworks/models/mixtral-8x7b-instruct": 0.50,    # Mixtral 8x7B
    "accounts/fireworks/models/llama-v3p1-8b-instruct": 0.20,   # Llama 3.1 8B
}

FIREWORKS_COST_PER_1M_OUTPUT = {
    "accounts/fireworks/models/llama-v3p3-70b-instruct": 0.90,
    "accounts/fireworks/models/qwen2p5-72b-instruct": 0.90,
    "accounts/fireworks/models/mixtral-8x7b-instruct": 0.50,
    "accounts/fireworks/models/llama-v3p1-8b-instruct": 0.20,
}


@dataclass
class FireworksModelInfo:
    """Information about a Fireworks model"""
    id: str
    name: str
    context_length: int
    supports_function_calling: bool
    supports_vision: bool
    cost_per_1m_input: float
    cost_per_1m_output: float


# Model catalog
FIREWORKS_MODELS = {
    "llama-v3p3-70b": FireworksModelInfo(
        id="accounts/fireworks/models/llama-v3p3-70b-instruct",
        name="Llama 3.3 70B Instruct",
        context_length=131072,
        supports_function_calling=True,
        supports_vision=False,
        cost_per_1m_input=0.90,
        cost_per_1m_output=0.90,
    ),
    "qwen2p5-72b": FireworksModelInfo(
        id="accounts/fireworks/models/qwen2p5-72b-instruct",
        name="Qwen 2.5 72B Instruct",
        context_length=32768,
        supports_function_calling=True,
        supports_vision=False,
        cost_per_1m_input=0.90,
        cost_per_1m_output=0.90,
    ),
    "mixtral-8x7b": FireworksModelInfo(
        id="accounts/fireworks/models/mixtral-8x7b-instruct",
        name="Mixtral 8x7B Instruct",
        context_length=32768,
        supports_function_calling=True,
        supports_vision=False,
        cost_per_1m_input=0.50,
        cost_per_1m_output=0.50,
    ),
    "llama-v3p1-8b": FireworksModelInfo(
        id="accounts/fireworks/models/llama-v3p1-8b-instruct",
        name="Llama 3.1 8B Instruct",
        context_length=131072,
        supports_function_calling=True,
        supports_vision=False,
        cost_per_1m_input=0.20,
        cost_per_1m_output=0.20,
    ),
}


class FireworksClient:
    """
    Client for Fireworks AI API.
    
    Provides fast inference for open-source models with streaming support.
    """
    
    def __init__(
        self,
        api_key: str = FIREWORKS_API_KEY,
        base_url: str = FIREWORKS_API_BASE,
        default_model: str = FIREWORKS_DEFAULT_MODEL,
    ):
        self.logger = logging.getLogger("FireworksClient")
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self._session = None
        
        # Statistics
        self._total_requests = 0
        self._total_tokens_input = 0
        self._total_tokens_output = 0
        self._total_cost = 0.0
    
    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession()
            except ImportError:
                raise RuntimeError("aiohttp package not installed. Install with: pip install aiohttp")
        return self._session
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = FIREWORKS_MAX_TOKENS,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response using Fireworks AI.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model ID (defaults to FIREWORKS_DEFAULT_MODEL)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            system_prompt: System prompt to prepend
            **kwargs: Additional parameters for Fireworks API
        
        Returns:
            Dictionary with response data:
                {
                    "content": str,
                    "model": str,
                    "tokens_input": int,
                    "tokens_output": int,
                    "cost": float,
                    "latency_ms": float,
                    "success": bool,
                    "error": Optional[str]
                }
        """
        start_time = time.time()
        model = model or self.default_model
        
        result = {
            "content": "",
            "model": model,
            "tokens_input": 0,
            "tokens_output": 0,
            "cost": 0.0,
            "latency_ms": 0.0,
            "success": False,
            "error": None,
        }
        
        if not self.api_key:
            result["error"] = "Fireworks API key not configured"
            self.logger.error(result["error"])
            return result
        
        try:
            import aiohttp
            
            # Build messages list
            api_messages = []
            if system_prompt:
                api_messages.append({"role": "system", "content": system_prompt})
            
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant", "system"):
                    api_messages.append({"role": role, "content": content})
            
            # API payload
            payload = {
                "model": model,
                "messages": api_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            session = await self._get_session()
            
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=FIREWORKS_TIMEOUT_SECONDS),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # Extract response
                    result["content"] = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    # Token usage
                    usage = data.get("usage", {})
                    result["tokens_input"] = usage.get("prompt_tokens", 0)
                    result["tokens_output"] = usage.get("completion_tokens", 0)
                    
                    # Calculate cost
                    cost_input = FIREWORKS_COST_PER_1M_INPUT.get(model, 0.90)
                    cost_output = FIREWORKS_COST_PER_1M_OUTPUT.get(model, 0.90)
                    result["cost"] = (
                        (result["tokens_input"] / 1_000_000) * cost_input +
                        (result["tokens_output"] / 1_000_000) * cost_output
                    )
                    
                    result["success"] = True
                    
                    # Update statistics
                    self._total_requests += 1
                    self._total_tokens_input += result["tokens_input"]
                    self._total_tokens_output += result["tokens_output"]
                    self._total_cost += result["cost"]
                    
                    self.logger.info(
                        f"[Fireworks] {model.split('/')[-1]} | "
                        f"{result['tokens_output']} tokens | "
                        f"${result['cost']:.4f} | "
                        f"{result['latency_ms']:.0f}ms"
                    )
                else:
                    error_text = await resp.text()
                    result["error"] = f"HTTP {resp.status}: {error_text}"
                    self.logger.error(f"[Fireworks] API error: {result['error']}")
        
        except asyncio.TimeoutError:
            result["error"] = "Request timeout"
            self.logger.error(f"[Fireworks] Timeout after {FIREWORKS_TIMEOUT_SECONDS}s")
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"[Fireworks] Error: {e}", exc_info=True)
        
        result["latency_ms"] = (time.time() - start_time) * 1000
        return result
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = FIREWORKS_MAX_TOKENS,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response using Fireworks AI.
        
        Yields:
            Text chunks as they are generated
        """
        model = model or self.default_model
        
        if not self.api_key:
            yield "[Error: Fireworks API key not configured]"
            return
        
        try:
            import aiohttp
            
            # Build messages
            api_messages = []
            if system_prompt:
                api_messages.append({"role": "system", "content": system_prompt})
            
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant", "system"):
                    api_messages.append({"role": role, "content": content})
            
            # API payload with streaming enabled
            payload = {
                "model": model,
                "messages": api_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                **kwargs
            }
            
            session = await self._get_session()
            
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=FIREWORKS_TIMEOUT_SECONDS),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    yield f"[Error: HTTP {resp.status}: {error_text}]"
                    return
                
                # Stream SSE response
                async for line in resp.content:
                    line_text = line.decode('utf-8').strip()
                    
                    if not line_text or line_text == "data: [DONE]":
                        continue
                    
                    if line_text.startswith("data: "):
                        try:
                            import json
                            data = json.loads(line_text[6:])
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        
        except asyncio.TimeoutError:
            yield f"[Error: Timeout after {FIREWORKS_TIMEOUT_SECONDS}s]"
        except Exception as e:
            yield f"[Error: {e}]"
            self.logger.error(f"[Fireworks] Streaming error: {e}", exc_info=True)
    
    async def health_check(self) -> bool:
        """Check if Fireworks API is accessible."""
        if not self.api_key:
            return False
        
        try:
            import aiohttp
            session = await self._get_session()
            
            # Try a minimal request
            payload = {
                "model": self.default_model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
            }
            
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                return resp.status == 200
        except Exception as e:
            self.logger.debug(f"[Fireworks] Health check failed: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self._total_requests,
            "total_tokens_input": self._total_tokens_input,
            "total_tokens_output": self._total_tokens_output,
            "total_cost": self._total_cost,
            "avg_tokens_per_request": (
                self._total_tokens_output / self._total_requests
                if self._total_requests > 0 else 0
            ),
        }
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None


# Module-level singleton
_client_instance: Optional[FireworksClient] = None


def get_fireworks_client() -> FireworksClient:
    """Get or create Fireworks client singleton."""
    global _client_instance
    if _client_instance is None:
        _client_instance = FireworksClient()
    return _client_instance
