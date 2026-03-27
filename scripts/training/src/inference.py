import argparse
import json
from pathlib import Path

import torch

from utils import load_qwen3_with_lora, load_yaml, repair_json_output


def build_prompt(user_input: str) -> str:
    return (
        "<|im_start|>system\n路由决策器<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--text",
        default="Project: AI legal co-pilot for SMB founders. Category: technology. Country: us. GoalUSD: 50000. DurationDays: 30.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    tokenizer, model = load_qwen3_with_lora(
        base_model_path=cfg["model"]["model_name_or_path"],
        lora_dir=str(Path(cfg["output"]["adapter_dir"]) / "adapter_model"),
        anchor_file=__file__,
    )

    prompt = build_prompt(args.text)
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
    generated = tokenizer.decode(gen_ids, skip_special_tokens=True)
    repaired_obj, repaired_str = repair_json_output(generated)

    print("Raw model output:\n")
    print(generated)
    print("\nRepaired JSON string:\n")
    print(repaired_str)

    if repaired_obj is None:
        print("\nFailed to parse JSON after repair.")
    else:
        print("\nParsed JSON object:\n")
        print(json.dumps(repaired_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
