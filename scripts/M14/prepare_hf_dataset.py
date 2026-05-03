import os
import json
from pathlib import Path
from typing import List

try:
    from datasets import Dataset
except Exception:
    raise SystemExit("Please install the 'datasets' package: pip install datasets")


def load_jsonl_dir(jsonl_dir: str) -> List[dict]:
    p = Path(jsonl_dir)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {jsonl_dir}")

    records: List[dict] = []
    for f in sorted(p.iterdir()):
        if f.is_file() and f.suffix.lower() in (".jsonl", ".json"):
            try:
                with f.open("r", encoding="utf-8") as fh:
                    if f.suffix.lower() == ".jsonl":
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                records.append(obj)
                            except Exception:
                                # try eval fallback
                                try:
                                    obj = eval(line)
                                    records.append(obj)
                                except Exception:
                                    continue
                    else:
                        data = json.load(fh)
                        if isinstance(data, list):
                            records.extend(data)
                        elif isinstance(data, dict):
                            records.append(data)
            except Exception:
                continue
    return records


def _save_dataset_as_jsonl(ds_obj, out_local: str):
    Path(out_local).mkdir(parents=True, exist_ok=True)
    # datasets.DatasetDict or Dataset
    try:
        # If it's a DatasetDict-like (has .items())
        for split, d in ds_obj.items():
            path = Path(out_local) / f"{split}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for rec in d:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # single Dataset
        path = Path(out_local) / "data.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for rec in ds_obj:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _anonymize_record(rec: dict) -> dict:
    """Recursively anonymize string fields (emails, phones, IPs, UUIDs, card numbers, simple passwords),
    recompute raw_hash from anonymized raw_content, and mark record as anonymized.

    This implementation always uses the local regex rules (preferred) because
    the project's trajectory_pipeline anonymizer is stricter about phone formats
    and may miss plain-digit mobile numbers common in logs.
    """
    import re
    import copy
    import hashlib

    # PII regexes (cover common cases including plain-digit Chinese mobiles)
    EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}")
    PLAIN_CN_MOBILE_RE = re.compile(r"\b1\d{10}\b")
    PHONE_SEP_RE = re.compile(r"(?:\+?[\d\s\-()]{7,20})")
    IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
    CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
    # naive password-like pattern: words containing @ or _ and digits/letters, or 'password' key values
    PASS_RE = re.compile(r"(?i)(password\s*[:=]\s*\S+|\b\S{6,}\d+\S*\b)")

    def anon_text(s: str) -> str:
        if not isinstance(s, str):
            return s
        s = EMAIL_RE.sub("[EMAIL]", s)
        # replace plain Chinese mobile numbers first
        s = PLAIN_CN_MOBILE_RE.sub("[PHONE]", s)
        # then general phone-like sequences
        s = PHONE_SEP_RE.sub(lambda m: "[PHONE]" if any(ch.isdigit() for ch in m.group(0)) else m.group(0), s)
        s = IP_RE.sub("[IP]", s)
        s = UUID_RE.sub("[ID]", s)
        s = CARD_RE.sub("[CARD]", s)
        s = PASS_RE.sub("[REDACTED]", s)
        return s

    def anon_obj(o):
        if isinstance(o, str):
            return anon_text(o)
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                # avoid anonymizing numeric values
                if isinstance(v, str):
                    out[k] = anon_text(v)
                elif isinstance(v, (dict, list)):
                    out[k] = anon_obj(v)
                else:
                    out[k] = v
            return out
        if isinstance(o, list):
            return [anon_obj(v) for v in o]
        return o

    # deep copy to avoid mutating input
    try:
        working = copy.deepcopy(rec)
    except Exception:
        working = dict(rec)

    # Recursively anonymize entire record structure
    try:
        working = anon_obj(working)
    except Exception:
        pass

    # Ensure raw_parsed (if present) is anonymized and used to build raw_content
    raw_content_summary = ""
    if isinstance(working.get("raw_parsed"), (dict, list)):
        try:
            raw_content_summary = json.dumps(working.get("raw_parsed"), ensure_ascii=False)
        except Exception:
            raw_content_summary = str(working.get("raw_parsed"))
    else:
        rc = working.get("raw_content") or ""
        if isinstance(rc, str):
            raw_content_summary = rc
        else:
            raw_content_summary = str(rc)

    # truncate to reasonable length
    raw_content_trunc = raw_content_summary if len(raw_content_summary) <= 1000 else raw_content_summary[:1000] + "..."
    working["raw_content"] = raw_content_trunc

    # recompute raw_hash from anonymized content
    try:
        working["raw_hash"] = hashlib.sha256(raw_content_trunc.encode("utf-8")).hexdigest()
    except Exception:
        pass

    # ensure anonymized flag
    working["anonymized"] = True

    return working


def clean_jsonl_dir(jsonl_dir: str):
    """Scan .jsonl files under jsonl_dir, remove leading comment/metadata lines and malformed JSONL lines,
    backup the original file as <name>.jsonl.bak (timestamped if needed), and overwrite with cleaned content.
    """
    from pathlib import Path
    import time

    p = Path(jsonl_dir)
    if not p.exists():
        return
    for f in sorted(p.rglob("*.jsonl")):
        # skip pretty files and dataset outputs
        if f.name.endswith(".pretty.jsonl"):
            continue
        if "hf_dataset" in f.parts:
            continue
        try:
            cleaned_lines = []
            with f.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if line.lstrip().startswith("//") or line.lstrip().startswith("#"):
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        cleaned = line
                        if cleaned.startswith(("//", "#")):
                            cleaned = cleaned.lstrip("/# ")
                        try:
                            obj = json.loads(cleaned)
                        except Exception:
                            # skip malformed line
                            continue
                    cleaned_lines.append(json.dumps(obj, ensure_ascii=False))

            if not cleaned_lines:
                # nothing to write
                continue

            # create backup name
            backup_path = f.with_name(f.name + ".bak")
            if backup_path.exists():
                ts = int(time.time())
                backup_path = f.with_name(f.name + f".bak.{ts}")
            # move original to backup
            try:
                f.rename(backup_path)
            except Exception:
                # if rename fails (same FS issue), write backup by copying content
                try:
                    import shutil
                    shutil.copy2(str(f), str(backup_path))
                except Exception:
                    pass

            # write cleaned content back to original path
            try:
                with f.open("w", encoding="utf-8") as out:
                    for ln in cleaned_lines:
                        out.write(ln + "\n")
            except Exception:
                # attempt to restore backup if write failed
                try:
                    if backup_path.exists() and not f.exists():
                        backup_path.rename(f)
                except Exception:
                    pass
        except Exception:
            continue


def save_hf_dataset(input_dir: str, out_dir: str):
    """Load logs from input_dir, map and save as HuggingFace dataset if datasets installed,
    otherwise write merged jsonl to out_dir/merged.jsonl
    """
    recs = load_jsonl_dir(input_dir)
    mapped = []
    for r in recs:
        try:
            m = map_record(r)
            mapped.append(m)
        except Exception:
            continue

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import Dataset
        ds = Dataset.from_list(mapped)
        ds.save_to_disk(str(outp))
        print(f"Saved HF dataset to {outp}")
    except Exception:
        # fallback to jsonl
        merged = outp / "merged.jsonl"
        with merged.open("w", encoding="utf-8") as fh:
            for item in mapped:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved merged jsonl to {merged}")


def main(jsonl_dir: str, out_local: str = "./hf_dataset", test_size: float = 0.01, push_to_hub: bool = False, repo_id: str = None, token: str = None, out_format: str = "arrow"):
    # Clean source jsonl files in-place (backup originals) to ensure downstream dataset is free of comment lines
    try:
        clean_jsonl_dir(jsonl_dir)
    except Exception:
        pass

    recs = load_jsonl_dir(jsonl_dir)
    print(f"Loaded {len(recs)} records from {jsonl_dir}")
    if not recs:
        raise SystemExit("No records found")

    # Anonymize loaded records to ensure no PII leaks into the HuggingFace dataset
    for i, r in enumerate(recs):
        try:
            recs[i] = _anonymize_record(r)
        except Exception:
            # keep original if anonymization fails for a record
            continue

    ds = Dataset.from_list(recs)
    if test_size > 0 and test_size < 1.0:
        ds = ds.train_test_split(test_size=test_size)
        # quick overlap check on raw_hash between splits
        try:
            train_hashes = {r.get('raw_hash') for r in ds['train']}
            test_hashes = {r.get('raw_hash') for r in ds['test']}
            overlap = train_hashes & test_hashes
            if overlap:
                print(f"WARNING: Found {len(overlap)} overlapping raw_hash values between train and test splits")
        except Exception:
            pass
    Path(out_local).mkdir(parents=True, exist_ok=True)

    # Save arrow/disk format if requested
    if out_format in ("arrow", "both"):
        ds.save_to_disk(out_local)
        print(f"Saved dataset to {out_local} (arrow)")

    # Save jsonl if requested
    if out_format in ("jsonl", "both"):
        jsonl_out_dir = os.path.join(out_local, "jsonl")
        _save_dataset_as_jsonl(ds, jsonl_out_dir)
        print(f"Saved jsonl splits to {jsonl_out_dir}")

    if push_to_hub:
        if not repo_id:
            raise SystemExit("repo_id required to push to HuggingFace Hub")
        # token can be passed or read from env
        hf_token = token or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            raise SystemExit("HUGGINGFACE_HUB_TOKEN not set in env and no token provided")
        ds.push_to_hub(repo_id, token=hf_token)
        print(f"Pushed dataset to {repo_id}")

    return ds


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Prepare HuggingFace dataset from jsonl shards")
    ap.add_argument("--jsonl-dir", required=True)
    ap.add_argument("--out-local", default="./hf_dataset")
    ap.add_argument("--test-size", type=float, default=0.01)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo-id")
    ap.add_argument("--token")
    ap.add_argument("--out-format", choices=["arrow", "jsonl", "both"], default="arrow", help="Output format for the dataset")
    args = ap.parse_args()
    main(args.jsonl_dir, out_local=args.out_local, test_size=args.test_size, push_to_hub=args.push, repo_id=args.repo_id, token=args.token, out_format=args.out_format)
