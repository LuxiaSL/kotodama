"""lm-evaluation-harness adapter for LuxiaBaseModel.

Wraps the existing model_loader and generate modules to provide
the TemplateLM interface that lm-eval expects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

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
        max_batch_tokens: int = 32768,
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
        self._max_batch_tokens = max_batch_tokens
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

    def _iter_padded_batches(
        self, items: list[tuple[int, list[int]]]
    ) -> "Iterator[tuple[list[tuple[int, list[int]]], torch.Tensor]]":
        """Yield (batch, input_ids) with right-padded, length-sorted batching.

        Right-padding is exact under causal attention: content positions
        never attend to trailing pads, and RoPE positions start at 0 for
        every row. (Left-padding would shift RoPE positions — never do that.)
        Batches respect both `batch_size` (rows) and `max_batch_tokens`
        (rows × padded width) so long-sequence batches can't OOM.
        """
        items = sorted(items, key=lambda t: len(t[1]), reverse=True)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        bs = max(1, self._batch_size)
        pos = 0
        while pos < len(items):
            # Pad width to a multiple of 64: fewer distinct shapes for the
            # kernel/compile caches; extra trailing pads don't affect content
            # logits under causal attention.
            width = max(64, ((len(items[pos][1]) + 63) // 64) * 64)
            rows = 1
            while (
                pos + rows < len(items)
                and rows < bs
                and (rows + 1) * width <= max(self._max_batch_tokens, width)
            ):
                rows += 1
            batch = items[pos : pos + rows]
            pos += rows

            input_ids = torch.full(
                (rows, width), pad_id, dtype=torch.long, device=self._device
            )
            for r, (_key, toks) in enumerate(batch):
                input_ids[r, : len(toks)] = torch.tensor(
                    toks, dtype=torch.long, device=self._device
                )
            yield batch, input_ids

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        **kwargs: Any,
    ) -> list[tuple[float, bool]]:
        n = len(requests)
        results: list[tuple[float, bool] | None] = [None] * n

        prepped: list[tuple[int, list[int]]] = []
        cont_lens: dict[int, int] = {}
        for idx, (_strings, ctx_toks, cont_toks) in enumerate(requests):
            full = ctx_toks + cont_toks
            if len(full) > self._max_length:
                full = full[-self._max_length :]
            prepped.append((idx, full))
            cont_lens[idx] = len(cont_toks)

        done = 0
        for batch, input_ids in self._iter_padded_batches(prepped):
            with torch.no_grad():
                logits = self.model(input_ids)["logits"]

            for r, (idx, toks) in enumerate(batch):
                seq_len = len(toks)
                cont_len = cont_lens[idx]

                # Logits at position t predict token t+1.
                # Continuation spans positions [seq_len - cont_len, seq_len),
                # so we need logits at [seq_len - cont_len - 1, seq_len - 1).
                start = seq_len - cont_len - 1
                end = seq_len - 1

                token_log_probs = F.log_softmax(
                    logits[r, start:end, :].float(), dim=-1
                )
                cont_ids = input_ids[r, start + 1 : end + 1]
                gathered = torch.gather(
                    token_log_probs, 1, cont_ids.unsqueeze(-1)
                ).squeeze(-1)

                total_ll = gathered.sum().item()
                greedy = bool(
                    (token_log_probs.argmax(-1) == cont_ids).all().item()
                )
                results[idx] = (total_ll, greedy)

            done += len(batch)
            if done % 2048 < len(batch):
                logger.info("loglikelihood: %d/%d requests", done, n)

        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]

    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        totals = [0.0] * len(requests)

        # Split every request into windows of max_length (stride
        # max_length - 1: token 0 of each window is "free"/context-only),
        # then batch windows across requests — each window is scored
        # independently, so this is exactly the sequential computation.
        windows: list[tuple[int, list[int]]] = []
        for ridx, request in enumerate(requests):
            token_ids = self.tok_encode(request.args[0])
            for start in range(0, len(token_ids), self._max_length - 1):
                window = token_ids[start : start + self._max_length]
                if len(window) < 2:
                    continue
                windows.append((ridx, window))

        for batch, input_ids in self._iter_padded_batches(windows):
            with torch.no_grad():
                logits = self.model(input_ids)["logits"]

            for r, (ridx, toks) in enumerate(batch):
                seq_len = len(toks)
                # Logits at position t predict token t+1.
                # Score tokens [1, N-1] using logits at [0, N-2].
                token_lps = F.log_softmax(
                    logits[r, : seq_len - 1, :].float(), dim=-1
                )
                target_ids = input_ids[r, 1:seq_len]
                gathered = torch.gather(
                    token_lps, 1, target_ids.unsqueeze(-1)
                ).squeeze(-1)
                totals[ridx] += gathered.sum().item()

        return totals

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
