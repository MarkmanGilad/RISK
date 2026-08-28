# EditorLM Local Agent Tools

This folder is the documented home for local scripts that Codex may run in this project. It is not a second workflow UI and does not contain credentials.

## Approval rule

Codex first explains the proposed processing or embedding scope and waits for the user's approval. A project skill may explicitly waive this approval requirement. Source Registration remains the exception: EditorLM performs that bounded, local preparation automatically when a source changes.

## Current state

Markdown and plain-text processing does not need a command: after approval, Codex uses normal workspace reading and file-edit approval behavior to create the documented summaries and corpus-navigation artifacts. No embedding command is available yet.

<!-- editorlm-managed-tools: v1 -->
## Managed-tool update rule

`extract-source.py` is an EditorLM-managed script. An EditorLM update may replace it with a corrected version; do not customize it inside a project. Keep project-specific instructions in skills or other project artifacts instead.

<!-- editorlm-html-extractor: v1 -->
### HTML and HTM local conversion

After the user explicitly approves an HTML/HTM source, run this command from the project root:

```powershell
python .editorlm/tools/extract-html.py --source sources/path/to/file.html
```

The tool accepts only one local source beneath `sources/`. It does not access the network or run browser code. It writes `.editorlm/processed/<source-id>/parsed-text.txt` and `location-map.json`, then updates `.editorlm/state/source-registry.json` and `.editorlm/state/processing-log.jsonl`. If it fails because its Python dependency is unavailable, stop and ask the user before changing the environment.

<!-- editorlm-local-source-extractor: v1 -->
### Local source conversion

After explicit user approval, use this one-source command for HTML/HTM, RTF, CSV, XLSX, XLS, selectable-text PDF, DOCX, or PPTX:

```powershell
python .editorlm/tools/extract-source.py --source sources/path/to/file.ext
```

The tool accepts only a relative path under `sources/`. It runs locally, never uses the network or a browser, preserves the original source, and creates `.editorlm/processed/<source-id>/parsed-text.txt` plus `location-map.json`. It updates `.editorlm/state/source-registry.json` and `.editorlm/state/processing-log.jsonl`. Read the generated `parsed-text.txt`, not the original binary source, in a text editor. Open original PDFs in the installed PDF viewer only when visual verification is needed. A PDF without selectable text reports a failure for the later OCR or image-vision workflow; do not retry it with OCR now. Image files have no extraction command: after approval, Codex vision may describe charts, diagrams, photos, layouts, and short labels, but must stop rather than transcribe a long text image until OCR exists.

## Future tools

- Local conversion support may be added for additional formats after it is approved.
- `embed-sources` may later create vectors only for an approved set of already processed sources.

Any future tool remains a local launch point: Codex invokes it in the terminal only after the user approves its scope, and it never exposes provider credentials in the project.
