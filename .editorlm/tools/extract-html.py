#!/usr/bin/env python
"""EditorLM local HTML/HTM extractor."""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

try:
    from bs4 import BeautifulSoup
except ImportError as error:
    raise SystemExit(
        "Beautiful Soup is required for HTML extraction. Ask before changing the Python environment."
    ) from error


BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre", "table"}
IGNORED_TAGS = ("script", "style", "noscript", "template")


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalized_source_path(value):
    value = value.replace("\\", "/")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 2 or path.parts[0] != "sources":
        raise ValueError("--source must be a relative file path beneath sources/.")
    if path.suffix.lower() not in {".html", ".htm"}:
        raise ValueError("--source must name an .html or .htm file.")
    return path.as_posix()


def source_id(relative_path):
    stem = PurePosixPath(relative_path).stem
    slug = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").lower() or "source"
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
    return f"{slug}--{digest}"


def clean_text(value):
    return re.sub(r"\s+", " ", value).strip()


def text_with_links(element):
    text = clean_text(element.get_text(" ", strip=True))
    links = []
    for anchor in element.find_all("a", href=True):
        label = clean_text(anchor.get_text(" ", strip=True)) or anchor["href"]
        links.append(f"[{label}]({anchor['href']})")
    if links:
        return f"{text} — Links: {', '.join(links)}"
    return text


def append_block(lines, entries, source_location, kind, text):
    if not text:
        return
    if lines:
        lines.append("")
    start = len(lines) + 1
    lines.extend(text.splitlines())
    entries.append({
        "sourceLocation": source_location,
        "kind": kind,
        "parsedTextLineStart": start,
        "parsedTextLineEnd": len(lines),
    })


def extract_html(source_path, relative_path):
    soup = BeautifulSoup(source_path.read_bytes(), "html.parser")
    for tag in soup.find_all(IGNORED_TAGS):
        tag.decompose()

    root = soup.body or soup
    lines = []
    entries = []
    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    if title:
        append_block(lines, entries, "title[1]", "title", f"# {title}")

    ordinals = {}
    for element in root.find_all(list(BLOCK_TAGS)):
        if element.find_parent(BLOCK_TAGS):
            continue
        tag_name = element.name.lower()
        ordinals[tag_name] = ordinals.get(tag_name, 0) + 1
        location = f"{tag_name}[{ordinals[tag_name]}]"
        if tag_name == "table":
            rows = element.find_all("tr")
            for row_number, row in enumerate(rows, start=1):
                cells = row.find_all(("th", "td"), recursive=False)
                values = [text_with_links(cell) for cell in cells]
                append_block(
                    lines,
                    entries,
                    f"{location}/row[{row_number}]",
                    "table-row",
                    " | ".join(values),
                )
            continue
        text = text_with_links(element)
        if tag_name.startswith("h") and len(tag_name) == 2 and tag_name[1].isdigit():
            text = f"{'#' * int(tag_name[1])} {text}"
        elif tag_name == "li":
            text = f"- {text}"
        elif tag_name == "blockquote":
            text = f"> {text}"
        append_block(lines, entries, location, tag_name, text)

    if not lines:
        fallback = clean_text(root.get_text(" ", strip=True))
        append_block(lines, entries, "document[1]", "document", fallback)
    parsed_text = "\n".join(lines).strip() + "\n"
    return parsed_text, entries


def read_registry(path):
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if parsed.get("schemaVersion") == 1 and isinstance(parsed.get("sources"), dict):
            return parsed
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {"schemaVersion": 1, "updatedAt": "1970-01-01T00:00:00Z", "sources": {}}


def write_registry_and_log(project_root, registry, entry):
    state = project_root / ".editorlm" / "state"
    state.mkdir(parents=True, exist_ok=True)
    registry["updatedAt"] = utc_now()
    (state / "source-registry.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    with (state / "processing-log.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Extract one approved local HTML/HTM source for EditorLM.")
    parser.add_argument("--project-root", default=".", help="EditorLM project root (defaults to current folder).")
    parser.add_argument("--source", required=True, help="Relative HTML/HTM source path beneath sources/.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    relative_path = normalized_source_path(args.source)
    source_path = (project_root / Path(*PurePosixPath(relative_path).parts)).resolve()
    try:
        source_path.relative_to((project_root / "sources").resolve())
    except ValueError as error:
        raise SystemExit("--source must resolve inside sources/.") from error
    if not source_path.is_file():
        raise SystemExit(f"Source does not exist: {relative_path}")

    registry_path = project_root / ".editorlm" / "state" / "source-registry.json"
    registry = read_registry(registry_path)
    now = utc_now()
    try:
        parsed_text, entries = extract_html(source_path, relative_path)
        identifier = source_id(relative_path)
        artifact_directory = project_root / ".editorlm" / "processed" / identifier
        artifact_directory.mkdir(parents=True, exist_ok=True)
        parsed_relative = f".editorlm/processed/{identifier}/parsed-text.txt"
        locations_relative = f".editorlm/processed/{identifier}/location-map.json"
        (project_root / Path(*PurePosixPath(parsed_relative).parts)).write_text(parsed_text, encoding="utf-8")
        location_map = {
            "schemaVersion": 1,
            "format": "html",
            "sourcePath": relative_path,
            "parsedTextPath": parsed_relative,
            "entries": entries,
        }
        (project_root / Path(*PurePosixPath(locations_relative).parts)).write_text(
            json.dumps(location_map, indent=2) + "\n", encoding="utf-8"
        )
        stats = source_path.stat()
        registry["sources"][relative_path] = {
            "format": "html",
            "lastProcessed": {
                "sourceModifiedAtMs": stats.st_mtime_ns / 1_000_000,
                "sourceSizeBytes": stats.st_size,
                "processedAt": now,
                "processorVersion": 1,
                "kind": "html-extracted",
                "parsedTextPath": parsed_relative,
                "locationMapPath": locations_relative,
            },
            "lastAttempt": {"at": now, "outcome": "processed"},
        }
        write_registry_and_log(project_root, registry, {
            "at": now,
            "operation": "html-extraction",
            "relativePath": relative_path,
            "outcome": "processed",
            "sourceSizeBytes": stats.st_size,
        })
        print(f"Extracted {relative_path} to {parsed_relative}")
    except Exception as error:
        registry["sources"][relative_path] = {
            "format": "html",
            "lastAttempt": {"at": now, "outcome": "failed", "error": str(error)},
        }
        write_registry_and_log(project_root, registry, {
            "at": now,
            "operation": "html-extraction",
            "relativePath": relative_path,
            "outcome": "failed",
            "error": str(error),
        })
        raise SystemExit(f"EditorLM HTML extraction failed: {error}") from error


if __name__ == "__main__":
    main()
