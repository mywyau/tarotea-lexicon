#!/usr/bin/env python3
"""Helpers for OpenAI Batch API workflows."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

TERMINAL_BATCH_STATUSES = {"completed", "failed", "cancelled", "expired"}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def submit_batch(
    client: OpenAI,
    *,
    input_jsonl: Path,
    endpoint: str,
    completion_window: str,
    metadata: dict[str, str] | None = None,
):
    with input_jsonl.open("rb") as fh:
        input_file = client.files.create(file=fh, purpose="batch")

    return client.batches.create(
        input_file_id=input_file.id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=metadata or {},
    )


def wait_for_batch(client: OpenAI, batch_id: str, poll_seconds: float):
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in TERMINAL_BATCH_STATUSES:
            return batch
        time.sleep(max(poll_seconds, 1.0))


def _file_content_to_text(content_obj: Any) -> str:
    if isinstance(content_obj, str):
        return content_obj
    if isinstance(content_obj, bytes):
        return content_obj.decode("utf-8")

    text_val = getattr(content_obj, "text", None)
    if isinstance(text_val, str):
        return text_val

    content_val = getattr(content_obj, "content", None)
    if isinstance(content_val, bytes):
        return content_val.decode("utf-8")
    if isinstance(content_val, str):
        return content_val

    read_fn = getattr(content_obj, "read", None)
    if callable(read_fn):
        data = read_fn()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        if isinstance(data, str):
            return data

    raise RuntimeError("Unable to decode batch output file content")


def load_batch_output_lines(client: OpenAI, file_id: str) -> list[dict[str, Any]]:
    text = _file_content_to_text(client.files.content(file_id))
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def collect_batch_results(client: OpenAI, batch) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    output_file_id = getattr(batch, "output_file_id", None)
    if output_file_id:
        output_rows = load_batch_output_lines(client, output_file_id)

    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id:
        error_rows = load_batch_output_lines(client, error_file_id)

    return output_rows, error_rows
