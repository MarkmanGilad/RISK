# Project Subject

Converse with the user in English.

<!-- editorlm-response-language: v1 -->
### Required response language

**You must write every response to the user in English**, including questions, explanations, approval requests, and the required next-step direction. Do not switch languages unless the user explicitly asks you to do so.

Help the user define or refine the active Project Subject before any Source Processing. Use normal Codex conversation and file-edit approval behavior; do not create a separate chat or workflow engine.

Activation: when the user says `Start Project Definition`, begin this skill immediately. Do not create `PROJECT_SUBJECT.md` yet. First ask exactly: **What are you writing, and what will be your sources?** Ask at most one focused follow-up question at a time, and only when the answer is not clear enough to direct Source Processing. Do not turn this step into planning, outlining, or a long questionnaire. When the subject is clear enough, briefly summarize your understanding and ask whether the user wants a draft. Create the first draft only after the user asks you to draft it.

Work with the user to create or revise `PROJECT_SUBJECT.md`. It should explain the intended output, central question or thesis, perspective or role, audience, constraints, and key issues to explore.

Do not treat the subject as agreed until the user explicitly approves it. After that approval, add this frontmatter marker at the top of the file so EditorLM can recognize it as the active Project Subject:

```md
---
editorlm_project_subject: agreed
---
```

Do not add, organize, process, or summarize source files in this skill.
