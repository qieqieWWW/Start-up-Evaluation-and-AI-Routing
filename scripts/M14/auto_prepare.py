#!/usr/bin/env python3
import sys
from pathlib import Path

# Resolve project root and ensure imports work regardless of CWD
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ensure local tools folder is importable
SYS_PATH_ADDED = False
tools_dir = Path(__file__).parent
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))
    SYS_PATH_ADDED = True

from trajectory_pipeline import pipeline
from prepare_hf_dataset import main as prepare_main
from prepare_hf_dataset import save_hf_dataset
from pathlib import Path as _Path
import json as _json


def _resolve_path(p: str) -> str:
    """Resolve a path string to an absolute path under project_root if relative."""
    if p is None:
        return ""
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((project_root / pp).resolve())


def _process_single_run(run_path: Path, normalized_base: Path, workers: int) -> list:
    """Run pipeline for one run directory and place outputs under normalized_base/run_name"""
    run_name = run_path.name
    run_normalized = normalized_base / run_name
    run_normalized.mkdir(parents=True, exist_ok=True)
    print(f"  - Normalizing run {run_name}: {run_path} -> {run_normalized}")
    out_files = pipeline(str(run_path), str(run_normalized), workers=workers)
    return out_files


def _export_dataset_to_jsonl(hf_dir: str, out_dir: str, splits=None, force=False):
    """Export a dataset saved with datasets.save_to_disk to jsonl per split.
    This is a lightweight embedded version of export_hf_to_jsonl.py used as fallback.
    """
    try:
        from datasets import load_from_disk
    except Exception:
        return

    hf_path = _Path(hf_dir)
    if not hf_path.exists():
        return

    ds = load_from_disk(str(hf_path))
    outp = _Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    available_splits = list(ds.keys()) if hasattr(ds, 'keys') else []
    use_splits = [s for s in (splits or available_splits) if s in available_splits]
    for split in use_splits:
        out_file = outp / f"{split}.jsonl"
        if out_file.exists() and not force:
            continue
        count = 0
        with out_file.open('w', encoding='utf-8') as fh:
            for rec in ds[split]:
                fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1


def main(log_dir: str,
         normalized_dir: str = "data/normalized_logs",
         out_local: str = "data/hf_dataset",
         test_size: float = 0.02,
         workers: int = 2,
         push: bool = False,
         repo_id: str = None,
         token: str = None):
    """Run normalization pipeline on log_dir (single run or logs folder) then prepare HuggingFace dataset.

    Behavior:
    - If log_dir is a run folder (contains files), process it as single run.
    - If log_dir is a logs folder containing multiple run subfolders, iterate each run and write per-run normalized outputs under normalized_dir/run_name.
    - If log_dir does not exist, fall back to project_root/logs and iterate its subdirs.
    """
    log_dir_resolved = Path(_resolve_path(str(log_dir)))
    normalized_dir_resolved = Path(_resolve_path(str(normalized_dir)))
    out_local_resolved = _resolve_path(str(out_local))

    all_out_files = []

    # Quick path: if the input directory contains structured analysis_report_*.json or other .json
    # treat these as ready-to-ingest HF source and call prepare_hf_dataset.main directly.
    json_candidates = sorted([p for p in log_dir_resolved.glob('analysis_report_*.json')] + [p for p in log_dir_resolved.glob('*.json')])
    if json_candidates:
        print(f"Detected JSON report files in {log_dir_resolved}, ingesting directly into HF dataset (skip normalization)")
        try:
            ds = prepare_main(str(log_dir_resolved), out_local=out_local_resolved, test_size=test_size, push_to_hub=push, repo_id=repo_id, token=token, out_format="both")
            print("Dataset ready at:", out_local_resolved)
            return ds
        except Exception as e:
            print("Direct ingest failed, will continue with normalization pipeline:", e)

    if log_dir_resolved.exists() and log_dir_resolved.is_dir():
        entries = list(log_dir_resolved.iterdir())
        has_files = any(e.is_file() for e in entries)
        subdirs = [e for e in entries if e.is_dir()]

        if has_files:
            print(f"Processing single run directory: {log_dir_resolved}")
            all_out_files.extend(_process_single_run(log_dir_resolved, normalized_dir_resolved, workers))
        elif subdirs:
            print(f"Processing multiple runs under: {log_dir_resolved}")
            for d in sorted(subdirs):
                all_out_files.extend(_process_single_run(d, normalized_dir_resolved, workers))
        else:
            # empty dir -> fallback
            fallback = project_root / "logs"
            if fallback.exists() and fallback.is_dir():
                print(f"Input dir empty; falling back to {fallback}")
                for d in sorted([p for p in fallback.iterdir() if p.is_dir()]):
                    all_out_files.extend(_process_single_run(d, normalized_dir_resolved, workers))
            else:
                raise FileNotFoundError(f"No logs found at {log_dir_resolved} and no fallback logs directory")
    else:
        # path not exists -> try project_root/logs
        fallback = project_root / "logs"
        if fallback.exists() and fallback.is_dir():
            print(f"Input path not found; using fallback logs directory: {fallback}")
            for d in sorted([p for p in fallback.iterdir() if p.is_dir()]):
                all_out_files.extend(_process_single_run(d, normalized_dir_resolved, workers))
        else:
            raise FileNotFoundError(f"No logs found at {log_dir_resolved} and no fallback logs directory")

    print(f"Normalization produced total {len(all_out_files)} files across runs")

    print(f"Preparing HuggingFace dataset from: {normalized_dir_resolved} -> {out_local_resolved} (test_size={test_size})")
    # request both arrow and jsonl outputs
    ds = prepare_main(str(normalized_dir_resolved), out_local=out_local_resolved, test_size=test_size, push_to_hub=push, repo_id=repo_id, token=token, out_format="both")
    print("Dataset ready at:", out_local_resolved)

    # ensure jsonl splits exist; if not, export from saved Arrow dataset as a fallback
    jsonl_dir = _Path(out_local_resolved) / "jsonl"
    if not jsonl_dir.exists() or not any(jsonl_dir.iterdir()):
        try:
            _export_dataset_to_jsonl(out_local_resolved, str(jsonl_dir), splits=['train', 'test'], force=False)
            print(f"Exported jsonl splits to {jsonl_dir} via fallback exporter")
        except Exception:
            pass
    return ds


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Normalize logs then create HuggingFace dataset")
    ap.add_argument("--log-dir", required=False, default="analysis_reports", help="Directory containing raw logs or JSON analysis reports (e.g. analysis_report_*.json)")
    ap.add_argument("--normalized-dir", default="data/normalized_logs")
    ap.add_argument("--out-local", default="data/hf_dataset")
    ap.add_argument("--test-size", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo-id")
    ap.add_argument("--token")
    args = ap.parse_args()

    main(args.log_dir, normalized_dir=args.normalized_dir, out_local=args.out_local, test_size=args.test_size, workers=args.workers, push=args.push, repo_id=args.repo_id, token=args.token)

    # CLI for direct HF dataset preparation
    ap = argparse.ArgumentParser(description="Auto prepare HF dataset from simulation logs")
    ap.add_argument("--log-dir", required=True, help="Directory containing jsonl/json logs from m16")
    ap.add_argument("--out-dir", required=True, help="Output directory to save HF dataset or merged jsonl")
    args = ap.parse_args()

    save_hf_dataset(args.log_dir, args.out_dir)