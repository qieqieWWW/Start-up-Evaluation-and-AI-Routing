import re
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Iterator, Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed

# Simple regex-based PII patterns
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}")
# tighten phone regex: require separators or international '+' or parentheses to avoid matching plain numeric fields
PHONE_RE = re.compile(r"(?:\+[\d\s\-()]{7,20}|\(?\d{2,4}\)?[-.\s]\d{3,4}[-.\s]\d{3,4})")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def _anonymize_text(text: str, name_map: Optional[Dict[str, str]] = None) -> str:
    s = EMAIL_RE.sub("[EMAIL]", text)
    s = PHONE_RE.sub("[PHONE]", s)
    s = IP_RE.sub("[IP]", s)
    s = UUID_RE.sub("[ID]", s)
    s = CARD_RE.sub("[CARD]", s)
    if name_map:
        for real, fake in name_map.items():
            s = s.replace(real, fake)
    return s


def _make_id(source: str, raw: Any) -> str:
    base = source + json.dumps(raw, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _anonymize_obj(obj: Any, name_map: Optional[Dict[str, str]] = None) -> Any:
    """Recursively anonymize only string values inside dicts/lists.

    Numeric values are left intact to avoid over-matching (e.g., timestamps, amounts).
    """
    if isinstance(obj, str):
        return _anonymize_text(obj, name_map)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, str):
                out[k] = _anonymize_text(v, name_map)
            elif isinstance(v, (dict, list)):
                out[k] = _anonymize_obj(v, name_map)
            else:
                out[k] = v
        return out
    if isinstance(obj, list):
        return [_anonymize_obj(v, name_map) for v in obj]
    return obj


def normalize_event(raw: Any, source: str = "unknown", name_map: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """
    Convert a raw log line/object into a standardized trajectory record.
    Standard schema (example):
    {
      "id": "<sha1>",
      "timestamp": "<iso or raw>",
      "agent": "agent_name",
      "event_type": "action|obs|message|state_update",
      "payload": { ... },
      "trajectory_state": { ... },
      "metadata": {"source_file": "...", "line": N},
      "anonymized": True,
      "raw_content": "<anonymized/truncated text>",
      "raw_hash": "<sha256>",
      "raw_parsed": { ... }  # optional parsed anonymized JSON
    }
    """
    try:
        # preserve original raw for raw_content/hash
        orig_raw = raw

        # parse raw into a Python object when possible
        parsed_raw_obj = None
        if isinstance(orig_raw, dict):
            parsed_raw_obj = orig_raw
        elif isinstance(orig_raw, str):
            try:
                parsed_raw_obj = json.loads(orig_raw)
            except Exception:
                parsed_raw_obj = None
        else:
            parsed_raw_obj = None

        # Normalize baseline data extraction
        if isinstance(orig_raw, str) and parsed_raw_obj is None:
            try:
                data = json.loads(orig_raw)
            except Exception:
                data = {"message": orig_raw}
        elif isinstance(orig_raw, dict):
            data = orig_raw
        elif parsed_raw_obj is not None:
            data = parsed_raw_obj if isinstance(parsed_raw_obj, dict) else {"value": parsed_raw_obj}
        else:
            data = {"value": orig_raw}

        ts = data.get("timestamp") or data.get("time") or data.get("ts") or None
        # if timestamp missing, try to use the source file modification time (mtime)
        if ts is None:
            try:
                srcp = Path(source)
                if srcp.exists():
                    ts = srcp.stat().st_mtime
            except Exception:
                ts = None
        agent = data.get("agent") or data.get("role") or data.get("actor") or "unknown"
        etype = data.get("type") or data.get("event") or "message"
        payload = data.get("payload") or data.get("data") or {}

        if isinstance(payload, str):
            payload = {"text": _anonymize_text(payload, name_map)}
        elif isinstance(payload, dict):
            for k, v in list(payload.items()):
                if isinstance(v, str):
                    payload[k] = _anonymize_text(v, name_map)

        # prepare raw_content and raw_parsed
        try:
            if isinstance(orig_raw, str):
                raw_text = orig_raw
            else:
                raw_text = json.dumps(orig_raw, ensure_ascii=False)
        except Exception:
            raw_text = str(orig_raw)

        raw_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        raw_parsed = None
        raw_content_summary = ""
        if parsed_raw_obj is not None:
            # anonymize parsed object only on string fields
            anon_parsed = _anonymize_obj(parsed_raw_obj, name_map)
            raw_parsed = anon_parsed
            try:
                raw_content_summary = json.dumps(anon_parsed, ensure_ascii=False)
            except Exception:
                raw_content_summary = str(anon_parsed)
        else:
            anon_raw = _anonymize_text(raw_text, name_map)
            raw_content_summary = anon_raw

        # truncate summary to reasonable length
        raw_content_trunc = raw_content_summary if len(raw_content_summary) <= 1000 else raw_content_summary[:1000] + "..."

        traj = {
            "id": _make_id(source, data),
            "timestamp": ts,
            "agent": agent,
            "event_type": etype,
            "payload": payload,
            "trajectory_state": data.get("state") or {},
            "metadata": {"source_file": source},
            "anonymized": True,
            "raw_content": raw_content_trunc,
            "raw_hash": raw_hash,
        }

        if raw_parsed is not None:
            traj["raw_parsed"] = raw_parsed

        return traj
    except Exception:
        return None


def iter_log_file(path: Path) -> Iterator[str]:
    """Yield lines from various file formats: .jsonl/.log/.txt or a .json (list or object)."""
    if path.suffix.lower() in {".jsonl", ".log", ".txt"}:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for rec in data:
                    yield json.dumps(rec, ensure_ascii=False)
            else:
                yield json.dumps(data, ensure_ascii=False)
    else:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def process_file(src: str, dst_dir: str, name_map: Optional[Dict[str, str]] = None) -> str:
    srcp = Path(src)
    outp = Path(dst_dir) / f"{srcp.stem}.jsonl"
    pretty_outp = Path(dst_dir) / f"{srcp.stem}.pretty.jsonl"
    outp.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with outp.open("w", encoding="utf-8") as out, pretty_outp.open("w", encoding="utf-8") as pretty:
        for idx, line in enumerate(iter_log_file(srcp), 1):
            traj = normalize_event(line, source=str(srcp), name_map=name_map)
            if traj:
                out.write(json.dumps(traj, ensure_ascii=False) + "\n")
                # pretty print for human reading
                pretty.write(json.dumps(traj, ensure_ascii=False, indent=2) + "\n\n")
                count += 1
    return str(outp)


def pipeline(input_dir: str, output_dir: str, workers: int = 4, name_map: Optional[Dict[str, str]] = None) -> List[str]:
    p = Path(input_dir)
    files = [str(x) for x in p.iterdir() if x.is_file()]
    os.makedirs(output_dir, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_file, f, output_dir, name_map): f for f in files}
        for fut in as_completed(futs):
            src = futs[fut]
            try:
                out = fut.result()
                results.append(out)
            except Exception as e:
                print("Error processing", src, e)
    return results


def redact_text(text: str) -> str:
    if not text:
        return text
    t = str(text)
    t = EMAIL_RE.sub("[EMAIL]", t)
    t = PHONE_RE.sub("[PHONE]", t)
    return t


def redact_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Redact common PII in a record in-place and return a copy."""
    out = {}
    for k, v in (record.items() if isinstance(record, dict) else []):
        try:
            if isinstance(v, str):
                out[k] = redact_text(v)
            else:
                out[k] = v
        except Exception:
            out[k] = v
    return out


def map_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Map an arbitrary simulation log record into HF-friendly fields.

    Handles richer schema found in analysis_result_*.json files by checking
    for user_query, final_summary, modules (M7/M6), and project_data.

    Returns a dict with at least: id, raw_content (string), raw_parsed (original dict), final (if present), raw_hash, anonymized
    """
    import hashlib
    if not isinstance(record, dict):
        # fallback to stringified content
        s = json.dumps(record, ensure_ascii=False) if record is not None else ""
        rc = redact_text(s)
        return {"id": hashlib.sha1(rc.encode('utf-8')).hexdigest(), "raw_content": rc, "raw_parsed": None, "final": None, "raw_hash": hashlib.sha256(rc.encode('utf-8')).hexdigest(), "anonymized": False}

    # Make a shallow copy and redact string fields
    rp: Dict[str, Any] = {}
    for k, v in record.items():
        if isinstance(v, str):
            rp[k] = redact_text(v)
        else:
            rp[k] = v

    # Helper: build a short summary from M6 latest observation
    def _obs_summary(obs: Optional[dict]) -> str:
        if not obs or not isinstance(obs, dict):
            return ""
        parts = []
        for key in ("goal_usd", "duration_days", "country", "combined_risk", "actual_funding_usd"):
            if key in obs:
                parts.append(f"{key}={obs.get(key)}")
        return "; ".join(parts)

    # raw_content: prefer explicit 'raw_content', then 'user_query', then M7 note / final_summary.fused_summary, then M6 latest observation, else dump
    raw_content = ""
    if rp.get("raw_content"):
        raw_content = rp.get("raw_content")
    elif rp.get("user_query"):
        raw_content = rp.get("user_query")
    else:
        # final_summary fused summary
        fs = rp.get("final_summary") or rp.get("final")
        if isinstance(fs, dict):
            fused = fs.get("fused_summary") or fs.get("fused_summary", "")
            if fused:
                raw_content = fused
            elif fs.get("actions"):
                try:
                    raw_content = json.dumps(fs.get("actions"), ensure_ascii=False)
                except Exception:
                    raw_content = str(fs.get("actions"))
        if not raw_content:
            # M7 LLM 推理 note
            mod = rp.get("modules") or {}
            m7llm = None
            try:
                m7llm = mod.get("M7_LLM推理") or mod.get("M7_LLM推理".encode('utf-8'), None)
            except Exception:
                m7llm = mod.get("M7_LLM推理") if isinstance(mod, dict) else None
            if isinstance(m7llm, dict) and m7llm.get("note"):
                raw_content = m7llm.get("note")
        if not raw_content:
            # M6 latest observation
            m6 = rp.get("modules", {}).get("M6_状态管理", {}) if isinstance(rp.get("modules"), dict) else None
            latest_obs = None
            if isinstance(m6, dict):
                latest = m6.get("latest_state") or {}
                latest_obs = latest.get("observation") if isinstance(latest, dict) else None
            if not latest_obs:
                # try trace
                trace = m6.get("trace") if isinstance(m6, dict) else None
                if isinstance(trace, list) and len(trace) > 0:
                    # pick first reset observation or last observation with observation
                    for ev in trace:
                        if isinstance(ev, dict) and ev.get("observation"):
                            latest_obs = ev.get("observation")
                            break
            if latest_obs:
                raw_content = _obs_summary(latest_obs)

    if not raw_content:
        try:
            raw_content = json.dumps(rp, ensure_ascii=False)
        except Exception:
            raw_content = str(rp)

    # final: try known locations including final_summary and common aliases
    final_val = None
    # check explicit keys
    for key in ("final", "result", "output", "final_summary"):
        val = rp.get(key)
        if val:
            final_val = val
            break

    # normalize final into text when possible (prefer fused_summary, then actions)
    final_text = None
    if isinstance(final_val, dict):
        if final_val.get("fused_summary"):
            final_text = final_val.get("fused_summary")
        elif final_val.get("actions"):
            try:
                final_text = json.dumps(final_val.get("actions"), ensure_ascii=False)
            except Exception:
                final_text = str(final_val.get("actions"))
        else:
            # fallback to stringify the final dict
            try:
                final_text = json.dumps(final_val, ensure_ascii=False)
            except Exception:
                final_text = str(final_val)
    elif final_val:
        final_text = str(final_val)

    # compute id and raw_hash
    try:
        rc = raw_content or ""
        raw_hash = hashlib.sha256(rc.encode("utf-8")).hexdigest()
    except Exception:
        raw_hash = ""

    rec_id = rp.get("session_id") or rp.get("session") or rp.get("project_id") or rp.get("id")
    if not rec_id:
        try:
            rec_id = hashlib.sha1((raw_hash + (rp.get("timestamp") or "")).encode("utf-8")).hexdigest()
        except Exception:
            rec_id = raw_hash

    metadata = {
        "project_data": rp.get("project_data"),
        "modules": {
            "M7": (rp.get("modules") or {}).get("M7_专家路由") if isinstance(rp.get("modules"), dict) else None,
            "M8": (rp.get("modules") or {}).get("M8_风险判定") if isinstance(rp.get("modules"), dict) else None,
        }
    }

    return {
        "id": rec_id,
        "raw_content": raw_content or "",
        "raw_parsed": rp,
        "final": final_text,
        "metadata": metadata,
        "raw_hash": raw_hash,
        "anonymized": False
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Trajectory normalization and anonymization pipeline")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out_files = pipeline(args.input_dir, args.output_dir, workers=args.workers)
    print("Wrote:", out_files)
