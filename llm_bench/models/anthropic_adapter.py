"""Anthropic streaming adapter — measures TTFT, TPS, token counts, cost."""

import time
import anthropic

from llm_bench.models.base import ModelAdapter, ModelResponse
from llm_bench.config import WARMUP_PROMPT, WARMUP_MAX_TOKENS


class AnthropicAdapter(ModelAdapter):
    def __init__(self, model_id: str, model_name: str, cost_input: float, cost_output: float):
        self.model_id = model_id
        self.model_name = model_name
        self.cost_input = cost_input    # $ per 1M input tokens
        self.cost_output = cost_output  # $ per 1M output tokens
        self.client = anthropic.Anthropic()

    async def warmup(self) -> None:
        """Send a trivial request to warm up the connection."""
        self.client.messages.create(
            model=self.model_id,
            max_tokens=WARMUP_MAX_TOKENS,
            messages=[{"role": "user", "content": WARMUP_PROMPT}],
        )

    async def generate(
        self, prompt: str, system: str = "", max_tokens: int = 4096
    ) -> ModelResponse:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        output_chunks: list[str] = []
        ttft_ms = 0.0
        first_token_received = False

        start_time = time.perf_counter()

        try:
            with self.client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    if not first_token_received:
                        ttft_ms = (time.perf_counter() - start_time) * 1000
                        first_token_received = True
                    output_chunks.append(text)

                final_message = stream.get_final_message()

            total_time_ms = (time.perf_counter() - start_time) * 1000
            output_text = "".join(output_chunks)

            input_tokens = final_message.usage.input_tokens
            output_tokens = final_message.usage.output_tokens

            # Calculate tokens/sec (output generation speed)
            gen_time_sec = (total_time_ms - ttft_ms) / 1000 if ttft_ms > 0 else total_time_ms / 1000
            tokens_per_sec = output_tokens / gen_time_sec if gen_time_sec > 0 else 0.0

            # Calculate cost
            cost = (input_tokens * self.cost_input / 1_000_000) + (
                output_tokens * self.cost_output / 1_000_000
            )

            return ModelResponse(
                model_name=self.model_name,
                prompt_id="",
                category="",
                output=output_text,
                ttft_ms=round(ttft_ms, 2),
                total_time_ms=round(total_time_ms, 2),
                tokens_per_sec=round(tokens_per_sec, 2),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=round(cost, 6),
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
        pass  # Anthropic client doesn't need cleanup
