import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from utils import load_qwen3_with_lora, load_yaml, repair_json_output, resolve_existing_path


def build_infer_prompt(user_input: str) -> str:
    return (
        "<|im_start|>system\n路由决策器<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    test_path = resolve_existing_path(
        str(Path(cfg["data"]["processed_dir"]) / "test.jsonl"), anchor_file=__file__
    )

    tokenizer, model = load_qwen3_with_lora(
        base_model_path=cfg["model"]["model_name_or_path"],
        lora_dir=str(Path(cfg["output"]["adapter_dir"]) / "adapter_model"),
        anchor_file=__file__,
    )

    rows = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    total = len(rows)
    tier_correct = 0
    json_success = 0
    l3_total = 0
    l3_hit = 0

    model.eval()
    for row in tqdm(rows, desc="Evaluating"):
        truth = json.loads(row["output"])
        truth_tier = truth.get("tier", "L1")

        prompt = build_infer_prompt(row["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
                temperature=float(cfg["inference"]["temperature"]),
                top_p=float(cfg["inference"]["top_p"]),
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        pred_obj, _ = repair_json_output(generated_text)
        if pred_obj is not None:
            json_success += 1
            pred_tier = pred_obj.get("tier", "")
            if pred_tier == truth_tier:
                tier_correct += 1
            if truth_tier == "L3":
                l3_total += 1
                if pred_tier == "L3":
                    l3_hit += 1
        else:
            if truth_tier == "L3":
                l3_total += 1

    tier_acc = tier_correct / total if total else 0.0
    json_rate = json_success / total if total else 0.0
    l3_recall = l3_hit / l3_total if l3_total else 0.0

    report = {
        "total": total,
        "tier_accuracy": round(tier_acc, 4),
        "json_parse_success_rate": round(json_rate, 4),
        "l3_recall": round(l3_recall, 4),
    }

    report_path = Path(cfg["output"]["adapter_dir"]) / "eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
