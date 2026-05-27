"""lm-evaluation-harness adapter for LuxiaBaseModel.

Wraps the existing model_loader and generate modules to provide
the TemplateLM interface that lm-eval expects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from transformers import AutoTokenizer

from src.eval.generate import generate as luxia_generate
from src.eval.model_loader import load_model

logger = logging.getLogger(__name__)

DEFAULT_ATTN_RES_CONFIG: dict[str, Any] = {
    "attn_res": True,
    "attn_res_n_blocks": 7,
    "attn_res_boundaries": [0, 3, 7, 12, 21, 25],
}


class LuxiaEvalLM(TemplateLM):
    """lm-eval adapter for LuxiaBaseModel checkpoints."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path = "configs/model.yaml",
        config_section: str = "proxy",
        attn_res_config: dict[str, Any] | None = None,
        device: str = "cuda:0",
        batch_size: int = 4,
        max_length: int = 4096,
        compile: bool = False,
    ) -> None:
        super().__init__()

        if attn_res_config is None:
            attn_res_config = DEFAULT_ATTN_RES_CONFIG

        self.model = load_model(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            config_section=config_section,
            attn_res_config=attn_res_config,
            device=device,
        )

        if compile:
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
            logger.info("Compilation registered (will compile on first forward)")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._device = torch.device(device)
        self._batch_size = batch_size
        self._max_length = max_length
        self._checkpoint_name = Path(checkpoint_path).stem

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "LuxiaEvalLM ready: %s (%.1fM params, bs=%d, device=%s)",
            self._checkpoint_name,
            param_count / 1e6,
            batch_size,
            device,
        )

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **kwargs: Any
    ) -> list[int]:
        if add_special_tokens is None:
            add_special_tokens = False
        return self.tokenizer.encode(
            string, add_special_tokens=add_special_tokens
        )

    def tok_decode(self, tokens: list[int], **kwargs: Any) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        **kwargs: Any,
    ) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []

        # Process one at a time — RoPE assigns positions from index 0, so
        # left-padding would give content tokens wrong positional encodings.
        # At 108M params, unbatched GPU inference is fast enough.
        for orig_idx, (_strings, ctx_toks, cont_toks) in enumerate(requests):
            full = ctx_toks + cont_toks
            if len(full) > self._max_length:
                full = full[-self._max_length :]
            cont_len = len(cont_toks)
            seq_len = len(full)

            input_ids = torch.tensor(
                [full], dtype=torch.long, device=self._device
            )

            with torch.no_grad():
                output = self.model(input_ids)
                logits = output["logits"].float()

            log_probs = F.log_softmax(logits, dim=-1)

            # Logits at position t predict token t+1
            # Continuation spans positions [seq_len - cont_len, seq_len)
            # So we need logits at [seq_len - cont_len - 1, seq_len - 1)
            start = seq_len - cont_len - 1
            end = seq_len - 1

            cont_ids = input_ids[0, start + 1 : end + 1]
            token_log_probs = log_probs[0, start:end, :]
            gathered = torch.gather(
                token_log_probs, 1, cont_ids.unsqueeze(-1)
            ).squeeze(-1)

            total_ll = gathered.sum().item()
            greedy = (token_log_probs.argmax(-1) == cont_ids).all().item()

            results.append((total_ll, greedy))

        return results

    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        results: list[float] = []

        for request in requests:
            text = request.args[0]
            token_ids = self.tok_encode(text)

            total_ll = 0.0

            # Process in windows of max_length
            for start in range(0, len(token_ids), self._max_length - 1):
                window = token_ids[start : start + self._max_length]
                if len(window) < 2:
                    continue

                input_ids = torch.tensor(
                    [window], dtype=torch.long, device=self._device
                )
                with torch.no_grad():
                    output = self.model(input_ids)
                    logits = output["logits"].float()

                log_probs = F.log_softmax(logits, dim=-1)

                # Logits at position t predict token t+1.
                # Score tokens [1, N-1] using logits at [0, N-2].
                # Token 0 in each window is "free" (context-only).
                target_ids = input_ids[0, 1:]
                token_lps = log_probs[0, :-1, :]
                gathered = torch.gather(
                    token_lps, 1, target_ids.unsqueeze(-1)
                ).squeeze(-1)
                total_ll += gathered.sum().item()

            results.append(total_ll)

        return results

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        results: list[str] = []

        for request in requests:
            context = request.args[0]
            gen_kwargs = request.args[1] if len(request.args) > 1 else {}

            until = gen_kwargs.get("until", [])
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            do_sample = gen_kwargs.get("do_sample", True)
            temperature = gen_kwargs.get("temperature", 0.7)

            if not do_sample or temperature == 0:
                temperature = 0.001

            ctx_ids = self.tok_encode(context)
            if len(ctx_ids) > self._max_length - max_gen:
                ctx_ids = ctx_ids[-(self._max_length - max_gen) :]

            input_tensor = torch.tensor(
                [ctx_ids], dtype=torch.long, device=self._device
            )

            output_ids = luxia_generate(
                    self.model,
                    input_tensor,
                    max_new_tokens=max_gen,
                    temperature=temperature,
                )

            continuation_ids = output_ids[0, len(ctx_ids) :].tolist()
            text = self.tok_decode(continuation_ids)

            for stop in until:
                if stop in text:
                    text = text[: text.index(stop)]

            results.append(text)

        return results
