"""Ollama streaming adapter — measures TTFT, TPS, token counts, and VRAM."""

import time
import json
import aiohttp

from llm_bench.models.base import ModelAdapter, ModelResponse
from llm_bench.config import OLLAMA_BASE_URL, WARMUP_PROMPT, WARMUP_MAX_TOKENS


class OllamaAdapter(ModelAdapter):
    def __init__(self, model_id: str, model_name: str):
        self.model_id = model_id
        self.model_name = model_name
        self.base_url = OLLAMA_BASE_URL
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def warmup(self) -> None:
        """Send a trivial request to ensure model is loaded."""
        session = await self._get_session()
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": WARMUP_PROMPT}],
            "stream": False,
            "options": {"num_predict": WARMUP_MAX_TOKENS},
        }
        async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
            await resp.read()

    async def get_vram_mb(self) -> float | None:
        """Query Ollama /api/ps for current VRAM usage."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/ps") as resp:
                data = await resp.json()
                for model_info in data.get("models", []):
                    if self.model_id in model_info.get("name", ""):
                        size_bytes = model_info.get("size_vram", 0)
                        return size_bytes / (1024 * 1024)
        except Exception:
            pass
        return None

    async def generate(
        self, prompt: str, system: str = "", max_tokens: int = 4096
    ) -> ModelResponse:
        session = await self._get_session()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": True,
            "options": {"num_predict": max_tokens},
        }

        output_chunks: list[str] = []
        ttft_ms = 0.0
        first_token_received = False
        eval_count = 0
        eval_duration_ns = 0
        prompt_eval_count = 0

        start_time = time.perf_counter()

        try:
            async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.content:
                    line = line.strip()
                    if not line:
                        continue
                    chunk = json.loads(line)

                    # Extract content from streaming chunk
                    msg = chunk.get("message", {})
                    content = msg.get("content", "")
                    if content and not first_token_received:
                        ttft_ms = (time.perf_counter() - start_time) * 1000
                        first_token_received = True
                    if content:
                        output_chunks.append(content)

                    # Final chunk has done=true with eval metrics
                    if chunk.get("done"):
                        eval_count = chunk.get("eval_count", 0)
                        eval_duration_ns = chunk.get("eval_duration", 0)
                        prompt_eval_count = chunk.get("prompt_eval_count", 0)

            total_time_ms = (time.perf_counter() - start_time) * 1000
            output_text = "".join(output_chunks)

            # Calculate tokens/sec from Ollama's own metrics
            if eval_duration_ns > 0:
                tokens_per_sec = eval_count / (eval_duration_ns / 1e9)
            elif total_time_ms > 0 and eval_count > 0:
                tokens_per_sec = eval_count / (total_time_ms / 1000)
            else:
                tokens_per_sec = 0.0

            memory_mb = await self.get_vram_mb()

            return ModelResponse(
                model_name=self.model_name,
                prompt_id="",
                category="",
                output=output_text,
                ttft_ms=round(ttft_ms, 2),
                total_time_ms=round(total_time_ms, 2),
                tokens_per_sec=round(tokens_per_sec, 2),
                input_tokens=prompt_eval_count,
                output_tokens=eval_count,
                cost_usd=0.0,
                memory_mb=round(memory_mb, 1) if memory_mb else None,
            )

        except Exception as e:
            total_time_ms = (time.perf_counter() - start_time) * 1000
            return ModelResponse(
                model_name=self.model_name,
                prompt_id="",
                category="",
                output="",
                ttft_ms=0,
                total_time_ms=round(total_time_ms, 2),
                tokens_per_sec=0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                error=str(e),
            )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
