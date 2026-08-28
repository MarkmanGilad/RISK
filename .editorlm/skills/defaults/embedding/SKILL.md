# Embedding and Semantic Retrieval

Converse with the user in English.

<!-- editorlm-response-language: v1 -->
### Required response language

**You must write every response to the user in English**, including questions, explanations, approval requests, and the required next-step direction. Do not switch languages unless the user explicitly asks you to do so.

Use normal Codex conversation and file-edit approval behavior. Do not embed a corpus automatically. After source summaries and corpus navigation artifacts are available, decide whether semantic retrieval would materially improve the current work. If it would, propose the smallest useful set of already processed sources and explain the expected benefit.

Do not run an embedding tool, make an embedding-provider request, or create vectors until the user explicitly approves. The user may request all sources or a specific scope. If the user declined embedding earlier, raise it again only if retrieval is now materially needed. A user may edit this skill to waive approval. When an approved local tool exists, read `.editorlm/tools/README.md` and run only the applicable documented command.
