"""Build a multimodal-shaped merged model by overlaying LoRA-tuned text weights
onto the multimodal base.

Why this exists: the LoRA-merged text-only checkpoint
(`merged/qwen35_9b/weasel/`) stores weights with the same `model.language_model.*`
namespace as the multimodal base (`models/Qwen3.5-9B/`), but lacks visual/mtp
components. vLLM 0.22.1's Qwen3_5ForConditionalGeneration is the only registered
arch and unconditionally builds the vision tower, so we need a checkpoint with
both: visual+mtp from base, language_model from our LoRA-merged text weights.

Output mirrors the base's shard layout so `model.safetensors.index.json` is
identical; we only overwrite the text keys inside each shard.
"""
import argparse
import json
import shutil
from pathlib import Path

import safetensors.torch as st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base", default="/data1/minsoo3.kim/weasel/models/Qwen3.5-9B"
    )
    ap.add_argument(
        "--text", default="/data1/minsoo3.kim/weasel/merged/qwen35_9b/weasel"
    )
    ap.add_argument(
        "--out", default="/data1/minsoo3.kim/weasel/merged/qwen35_9b/weasel_mm"
    )
    args = ap.parse_args()

    base = Path(args.base)
    text = Path(args.text)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base_idx = json.loads((base / "model.safetensors.index.json").read_text())
    text_idx = json.loads((text / "model.safetensors.index.json").read_text())

    text_shards: dict[str, dict[str, "object"]] = {}
    for shard in sorted(set(text_idx["weight_map"].values())):
        print(f"[overlay] loading text shard {shard}")
        text_shards[shard] = st.load_file(str(text / shard))

    text_lookup = {
        k: text_shards[text_idx["weight_map"][k]][k]
        for k in text_idx["weight_map"]
    }
    print(f"[overlay] text weights loaded: {len(text_lookup)} tensors")

    for shard in sorted(set(base_idx["weight_map"].values())):
        print(f"[overlay] processing base shard {shard}")
        tensors = st.load_file(str(base / shard))
        overlaid = 0
        for k in list(tensors.keys()):
            if k in text_lookup:
                tensors[k] = text_lookup[k]
                overlaid += 1
        print(
            f"[overlay]   {shard}: overlaid {overlaid}/{len(tensors)} tensors"
        )
        st.save_file(tensors, str(out / shard), metadata={"format": "pt"})

    for fname in [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "merges.txt",
        "vocab.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ]:
        src = base / fname
        if src.exists():
            shutil.copy2(src, out / fname)
            print(f"[overlay] copied {fname} from base")

    text_template = text / "chat_template.jinja"
    if text_template.exists():
        shutil.copy2(text_template, out / "chat_template.jinja")
        print("[overlay] overrode chat_template.jinja from text checkpoint")

    print(f"[overlay] done → {out}")


if __name__ == "__main__":
    main()
