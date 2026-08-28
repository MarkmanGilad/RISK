# Source Processing

Converse with the user in English.

<!-- editorlm-response-language: v1 -->
### Required response language

**You must write every response to the user in English**, including questions, explanations, approval requests, and the required next-step direction. Do not switch languages unless the user explicitly asks you to do so.

## Activation

When the user says `Assess Source Processing`, follow this skill. If `.editorlm/knowledge/corpus/CORPUS_MAP.md` already contains `editorlm_source_processing: complete`, do not process again; direct the user to `Review Evidence Gaps`.

## Workflow

1. Read `PROJECT_SUBJECT.md`, the source inventory, and the corpus map. Assess the complete registered corpus by relevance, using the subject, source paths, and filenames. Do not include or exclude a source merely because of its format. Recommend the smallest useful scope; include unclear filenames so they can be read.
2. Wait for explicit approval before reading selected source content, creating summaries, running an extractor, calling a provider, or changing the environment. If processing was declined, ask again only when it becomes materially useful.
3. For each approved source, use normal reading for Markdown/plain text. For HTML/HTM, RTF, CSV, XLSX, XLS, selectable-text PDF, DOCX, or PPTX, first follow the documented local command in `.editorlm/tools/README.md`. Read the resulting parsed text, not an original binary file.
4. Create or refresh: a one-to-three-sentence corpus overview entry, `GENERAL_SUMMARY.md`, `ACTIVE_CONTEXT_SUMMARY.md`, and source links in `CORPUS_MAP.md`. For images, use approved vision only for non-text content; stop on long or dense text images until OCR exists.
5. When the approved scope is complete, document failures, exclusions, or postponements. Then add the required completion marker at the top of `CORPUS_MAP.md` before reporting completion:

```md
---
editorlm_source_processing: complete
---
```

## Boundaries

Do not install packages, use the web, download material, call a provider, or invent a processing command without separate user approval. Do not add the completion marker after assessment or a partial run.

<!-- editorlm-agent-directed-processing: v1 -->
## Agent-directed processing

Assess the corpus and recommend the smallest useful processing scope. Wait for explicit user approval before creating summaries, calling a provider, or running a local processing tool. A user may edit this skill to waive the approval requirement. If the user declines, ask again only when processing becomes materially relevant.

<!-- editorlm-processing-assessment-safety: v1 -->
## Assessment safety boundary

When the user asks to assess Source Processing, assess only. You may inspect the source inventory, registry, existing project artifacts, and documented tool availability. Do **not** install packages or dependencies, run `pip`, `npm`, or another package manager, extract source content, create generated files, call a provider, or run an undocumented tool. First present the recommended scope and wait for the user's explicit approval. Approval of a processing scope does not authorize dependency installation; ask separately before any installation or environment change.

<!-- editorlm-readable-text-processing: v3 -->
## Readable-text processing

For an initial corpus run, assess the complete registered corpus using the Project Subject, user instructions, source paths, and filenames. Do not include or exclude a source merely because of its format. If a filename or path does not reveal whether the source is relevant, include it in the proposed scope: read it and then create its normal summaries. The user may approve all, exclude selected sources or folders, or approve a narrow scope. Later runs normally cover only new or changed sources.

During the assessment, identify sources in the proposed scope that need local source conversion before reading: HTML/HTM, RTF, CSV, Excel, selectable-text PDF, DOCX, and PPTX. After approval, run only the documented local extractor for each selected source, then read the parsed text and create its normal summaries. Images may enter through the separate approved vision rule. Do not silently omit files, install a dependency, or process an unextracted special format as though it were readable text.

For every approved Markdown or plain-text source, use normal Codex file reading and file-edit approval behavior to create or refresh three knowledge levels: (A) a one-to-three-sentence neutral overview entry and topic labels in `.editorlm/knowledge/corpus/CORPUS_OVERVIEW.md`; (B) the fuller neutral `GENERAL_SUMMARY.md`; and (C) the high-recall active-context summary. Update `CORPUS_MAP.md` with links to the original and generated artifacts. Do not look for, create, or claim to run a `process-sources` command. Special formats enter this flow after local source conversion.

<!-- editorlm-html-extraction: v1 -->
## HTML and HTM local conversion

HTML and HTM sources can enter the source-processing workflow after local extraction. During assessment, identify the registered HTML/HTM files that appear useful and recommend the smallest useful extraction scope. Do not run extraction during assessment. After the user explicitly approves a file or scope, read `.editorlm/tools/README.md` and run the documented `extract-html.py` command only for that approved source.

The tool reads local static HTML only: it does not browse, download pages, follow links, execute JavaScript, call a provider, or change the original source. It creates local parsed text and a location map under `.editorlm/processed/` and records the result in the existing registry. If it reports a failure, tell the user which source failed and why; do not install packages or substitute web content without separate approval.

After successful extraction, use the parsed text and location map to create the same Level A overview, Level B general summary, Level C context summary, and `CORPUS_MAP.md` links required for readable text. Preserve the original HTML as the evidence source.

<!-- editorlm-local-source-conversion: v1 -->
## Local source conversion

For an approved HTML/HTM, RTF, CSV, XLSX, XLS, selectable-text PDF, DOCX, or PPTX source, use the documented command in `.editorlm/tools/README.md`: `python .editorlm/tools/extract-source.py --source <source path>`. Run it only for the explicitly approved file or scope, one source at a time. It creates local parsed text and a location map, updates the registry/log, and never changes the original. Do not run `pip`, download a parser, use a browser, or call a provider as part of extraction.

Do not try to open an original PDF, DOCX, PPTX, spreadsheet, or other binary source as a text-editor file. After successful local extraction, read the resulting `.editorlm/processed/<source-id>/parsed-text.txt` and location map to create the Level A overview, Level B general summary, Level C context summary, and `CORPUS_MAP.md` links. Open an original PDF only with the installed PDF viewer when visual verification is needed.

If a PDF has no selectable text, report that it needs the later OCR or image-vision workflow. Do not attempt OCR now.

For an approved image source, use Codex vision to describe visible non-textual content such as a chart, diagram, photograph, layout, or short label. Clearly mark the saved description and any related corpus-map entry as AI-derived visual interpretation. If the image is primarily long or dense text, stop: explain that dedicated OCR is not implemented yet, do not try to transcribe it, and do not guess missing text. A later OCR stage may use vision only to help when OCR output remains unclear.

<!-- editorlm-unified-source-scope: v1 -->
## One relevance rule for every source format

Apply one source-selection rule to the complete registered corpus. File format does not decide whether a source belongs in the current processing scope. Use the Project Subject, the user's request, paths, and filenames to recommend relevant sources and to explain any source you recommend postponing as clearly unrelated. When the name/path does not make relevance clear, include that source: use its local extractor first if needed, then read it and create the normal knowledge summaries. This rule supersedes any older instruction to process all Markdown/plain-text files by default or to treat special formats as automatically outside the first corpus run.
