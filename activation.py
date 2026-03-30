"""Residual stream activation extraction via TransformerLens.

Loads the unquantized model and captures intermediate activations
at configured hook points. This is the interpretability inference
path — separate from Ollama's operational inference.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformer_lens import HookedTransformer


@dataclass
class ActivationResult:
    """Container for a single prompt's activation extraction."""

    prompt: str
    hook_point: str
    activations: torch.Tensor  # shape: (seq_len, hidden_dim)
    tokens: list[str] = field(default_factory=list)


class ActivationExtractor:
    """Extracts residual stream activations from a HookedTransformer.

    Designed to be instantiated once and reused across many prompts.
    Model stays loaded on device between calls — only the forward
    pass is repeated per prompt.
    """

    def __init__(
        self,
        model_name: str,
        hook_point: str,
        device: str = "cuda",
        dtype: str = "float16",
        max_seq_len: int = 512,
    ):
        self.model_name = model_name
        self.hook_point = hook_point
        self.device = device
        self.max_seq_len = max_seq_len

        torch_dtype = getattr(torch, dtype, torch.float16)

        self.model: Optional[HookedTransformer] = None
        self._torch_dtype = torch_dtype

    def load(self) -> None:
        """Load model onto device. Call once at startup."""
        self.model = HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device,
            dtype=self._torch_dtype,
        )
        self.model.eval()

    def extract(self, prompt: str) -> ActivationResult:
        """Run a single prompt and capture activations at the hook point.

        Returns the residual stream tensor at the configured layer,
        truncated to max_seq_len.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        tokens = self.model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > self.max_seq_len:
            tokens = tokens[:, : self.max_seq_len]

        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=[self.hook_point],
            )

        # cache[hook_point] shape: (batch=1, seq_len, hidden_dim)
        activations = cache[self.hook_point].squeeze(0).detach()
        str_tokens = self.model.to_str_tokens(tokens.squeeze(0))

        return ActivationResult(
            prompt=prompt,
            hook_point=self.hook_point,
            activations=activations,
            tokens=str_tokens,
        )

    def extract_batch(self, prompts: list[str]) -> list[ActivationResult]:
        """Extract activations for multiple prompts sequentially.

        Sequential rather than batched to stay within VRAM budget
        on consumer GPUs (12GB RX 6700 XT).
        """
        results = []
        for prompt in prompts:
            results.append(self.extract(prompt))
            # Explicit cache clear between prompts
            torch.cuda.empty_cache()
        return results

    def unload(self) -> None:
        """Release model from device memory."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
