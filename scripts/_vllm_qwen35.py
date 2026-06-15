"""vLLM CLI shim for Qwen3.5 text-only standalone models.

vLLM 0.22.1 ships a Qwen3_5ForCausalLM class but only registers the multimodal
Qwen3_5ForConditionalGeneration. The LoRA-merged text-only checkpoints saved by
this repo write weights with an extra `language_model.` segment (e.g.
`model.language_model.embed_tokens.weight`) because the multimodal config was
in scope at merge time. vLLM's standalone Qwen3_5ForCausalLM expects them at
`model.*`. Strip that segment as weights stream in, then register the subclass
under the arch name vLLM looks up.

Usage mirrors `vllm`:
  python scripts/_vllm_qwen35.py serve <model> --port 8002 ...
"""
import sys
from collections.abc import Iterable

import torch
from transformers import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


class _Qwen35TextConfig(PretrainedConfig):
    model_type = "qwen3_5_text"


CONFIG_MAPPING.register("qwen3_5_text", _Qwen35TextConfig, exist_ok=True)

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM as _Qwen3_5ForCausalLM,
)
from vllm.model_executor.models.registry import ModelRegistry


class Qwen3_5ForCausalLMTextStandalone(_Qwen3_5ForCausalLM):
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        def remap():
            for name, w in weights:
                if name.startswith("model.language_model."):
                    name = "model." + name[len("model.language_model.") :]
                yield name, w

        return super().load_weights(remap())


ModelRegistry.register_model(
    "Qwen3_5ForCausalLM",
    Qwen3_5ForCausalLMTextStandalone,
)


if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    sys.argv[0] = "vllm"
    sys.exit(main())
