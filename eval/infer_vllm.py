"""Resumable greedy vLLM evaluation runner for five remote sensing VQA benchmarks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.prompts import DEFAULT_IMAGE_LONG_EDGE, format_prompt, load_image


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vLLM evaluation on a GeOPD benchmark manifest.")
    parser.add_argument("--model-path", type=str, required=True, help="Local path or Hugging Face model ID.")
    parser.add_argument("--manifest", type=Path, required=True, help="Benchmark JSONL manifest.")
    parser.add_argument("--output", type=Path, required=True, help="Output predictions JSONL path.")
    parser.add_argument("--image-root", type=Path, default=None, help="Optional root for relative image paths.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--image-long-edge", type=int, default=DEFAULT_IMAGE_LONG_EDGE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N pending records if > 0.")
    args = parser.parse_args()

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    rows = load_jsonl(args.manifest)
    existing = load_jsonl(args.output)
    done_uids = {row["uid"] for row in existing if row.get("status") == "success"}
    pending = [row for row in rows if row["uid"] not in done_uids]
    if args.limit > 0:
        pending = pending[: args.limit]

    if not pending:
        print(f"All {len(rows)} records already completed in {args.output}.")
        return

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        min_pixels=65536,
        max_pixels=1048576,
    )
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        seed=args.seed,
        limit_mm_per_prompt={"image": 4, "video": 0},
        mm_processor_kwargs={"min_pixels": 65536, "max_pixels": 1048576},
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            requests = []
            for row in batch:
                images = []
                for raw_path in row["images"]:
                    img_path = Path(raw_path)
                    if not img_path.is_absolute() and args.image_root is not None:
                        img_path = args.image_root / img_path
                    images.append(load_image(img_path, max_long_edge=args.image_long_edge))

                content = [{"type": "image", "image": im} for im in images]
                content.append({"type": "text", "text": format_prompt(row)})
                prompt = processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                requests.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": images if len(images) > 1 else images[0]},
                    }
                )

            t0 = time.time()
            outputs = llm.generate(requests, sampling, use_tqdm=False)
            elapsed = time.time() - t0

            for row, out in zip(batch, outputs):
                prediction = out.outputs[0].text
                record = {
                    "uid": row["uid"],
                    "dataset": row["dataset"],
                    "task": row["task"],
                    "status": "success",
                    "prediction": prediction,
                    "output_tokens": len(out.outputs[0].token_ids),
                    "batch_seconds": round(elapsed, 3),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

            completed = len(done_uids) + min(start + len(batch), len(pending))
            print(f"[{completed}/{len(rows)}] evaluated batch ({elapsed:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
