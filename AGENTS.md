# Code style

## Plan reviews

- Treat requests to review, discuss, critique, or assess a plan as strictly read-only: do not change files or write code.
- Implement only after the user gives a specific, explicit instruction to make the requested change; a review request is never implementation authorization.

## Grill Me: initial plan alignment

- **The "Grill Me" technique.** When the user asks to start a brand-new plan — a fresh, not-yet-designed piece of work — interview them relentlessly before producing it (and before entering plan mode): walk each branch of the design tree and resolve dependencies one by one until reaching real alignment, rather than a shallow round of questions once. Let the design's depth decide how many questions that takes.
- **When it does not apply:** correcting, adjusting, or continuing an existing plan; or fixing code against a plan already agreed on. In those cases, proceed normally without running the interview.

## Tests

- Use the project test environment at `C:\\venvs\\ai-rl`. Do not use
  `C:\\Users\\Gilad\\venvs\\ai-rl` — it's a separate, incomplete venv missing
  `svg.path`, which spuriously fails `test_game_loop.py`/`test_ui.py`.
- From the repository root, run the full suite with:
  `& "C:\\venvs\\ai-rl\\Scripts\\python.exe" -m pytest Temp/tests -q`
- For focused coverage, replace `Temp/tests` with the relevant test file(s),
  for example `Temp/tests/test_dueling_dqn.py Temp/tests/test_agents.py`.

- Prefer simple code over complicated code. Default to the smallest, most direct implementation that solves the problem in front of you.
- Do not wrap compact expressions just to meet an 80-character limit. Keep them on one line when readable; split only truly long signatures or calls into two or three sensible rows.
- Before writing new code, check whether an existing function/method already does what you need. Don't duplicate logic that exists elsewhere.
- Don't use static methods unless there's no other option, or an instance method would be noticeably more complicated to use. Default to instance methods.
- Don't create dataclasses unless you ask first.
- After every code change update the .md docs in Docs/ folder.
- Start documentation lookup with `Docs/Content.md`; it indexes the active
  documents by subsystem and points to the relevant current-code reference.
- Read the needed Docs file before making any changes so you can see if there is a class or function that can be used and to understand the big picture.
- Before writing or running tests, read `Docs/Testing.md` — it maps which `Temp/tests/*.py` file covers which subsystem, the shared `conftest.py` fixtures, and the conventions to mirror when adding new tests. Don't add a new test file without checking whether an existing one already covers that subsystem.
- After every code or doc change, add a dated entry to `Docs/ChangeLog.md` (newest entry on top, see its own "Convention for new entries" section) summarizing what changed and why, with the files touched. This is how another agent or session — possibly working in parallel — finds out what happened since it last looked, without re-reading every diff.

<!-- editorlm-project-subject-start: v5 -->
## EditorLM Project Subject start

On `Start Project Definition`, read and follow `.editorlm/skills/defaults/project-context/SKILL.md`. Do not start source processing until the user approves `PROJECT_SUBJECT.md`.

<!-- editorlm-agent-workflow: v5 -->
## Big-picture workflow

Read `.editorlm/PROJECT_MAP.md` before source-grounded work. Use normal Codex conversation and file-edit approval behavior; EditorLM is not a second chat, editor, or planner.

1. **Project definition:** On `Start Project Definition`, use `skills/defaults/project-context/SKILL.md`.
2. **Source processing:** On `Assess Source Processing`, use `skills/defaults/source-processing/SKILL.md`. The user adds originals under `sources/`; EditorLM performs only automatic local registration.
3. **Evidence review:** On `Review Evidence Gaps`, use `skills/defaults/evidence-review/SKILL.md`.
4. **Embedding:** On `Assess Embedding`, use `skills/defaults/embedding/SKILL.md`. Never embed automatically.
5. **Plan, draft, review, verify:** Use the matching default skill. Before proposing `PLAN.md` or writing, complete evidence review. Never search, download, or use external material without explicit permission.

Keep user skills in `.editorlm/skills/user/`. Do not edit managed defaults or tools during project work.

EditorLM-managed scripts under `.editorlm/tools/` are product tools. If one appears defective, explain the issue and request an EditorLM product update; do not patch it inside the project.

## User-facing workflow language

Never expose EditorLM's internal implementation stages or stage numbers to the user. Describe the work in terms the user can act on: **source processing**, **local source conversion**, **visual description**, **summarization**, or **embedding**.

## Next-step rule

At the end of every response, state one clear next action the user can take.

<!-- editorlm-managed-tools: v1 -->
## EditorLM-managed tools

Do not edit EditorLM-managed scripts under `.editorlm/tools/`, including `.editorlm/tools/extract-source.py`, during project work. They are refreshed by EditorLM updates and are not user project artifacts. If a script appears defective, explain the issue, preserve the source, and ask the user whether they want an EditorLM product update; do not patch the script inside the project.

<!-- editorlm-user-skills: v1 -->
## User skills

Put user-created or user-customized skills in `.editorlm/skills/user/<skill-name>/SKILL.md`. Preserve those files during EditorLM updates and workflow-default restoration. Default skills directly under `.editorlm/skills/defaults/` are EditorLM workflow defaults; do not edit them during normal project work.

<!-- editorlm-work-log: v2 -->
## Work log

Read `.editorlm/WORK_LOG.md` when resuming project work. Whenever you create or update any project file, add a short human summary to its EditorLM fallback entry (or append one if no entry exists): Done; Decisions/approvals; Exceptions or open gaps; and Next. Do not log chat replies that do not change a file.

<!-- editorlm-canonical-status: v1 -->
## Determining current project status

Do not infer the current workflow status from `.editorlm/WORK_LOG.md`. The work log is only a concise handoff and audit record; it can be incomplete or delayed.

Before proposing a next workflow action, inspect the canonical project artifacts: `PROJECT_SUBJECT.md` for subject approval; `.editorlm/knowledge/corpus/CORPUS_MAP.md` for the exact `editorlm_source_processing: complete` marker and links to knowledge artifacts; and the relevant plan or draft files for later work. If the corpus map has the completion marker, do not tell the user to run Source Processing again. Continue with `Review Evidence Gaps` instead.

<!-- editorlm-project-subject-start: v4 -->
## EditorLM Project Subject start

When the user says `Start Project Definition`, immediately read and follow `.editorlm/skills/defaults/project-context/SKILL.md`. Begin by asking exactly: **What are you writing, and what will be your sources?** Ask at most one focused follow-up question at a time, and only if the answer is not clear enough to direct Source Processing. Do not turn this step into planning, outlining, or a long questionnaire. When the subject is clear enough, briefly summarize it and ask whether the user wants a draft. Do not create `PROJECT_SUBJECT.md` until the user has provided enough context and asks for a draft. Do not wait for the user to open or explain the skill file.

<!-- editorlm-agent-workflow: v4 -->
## Big-picture workflow

EditorLM is the local project and source-knowledge layer; it is not a second chat, editor, planner, or file explorer. Use normal Codex conversation, workspace tools, and file-edit approval behavior. Read `.editorlm/PROJECT_MAP.md` before source-grounded work.

EditorLM-managed scripts under `.editorlm/tools/`, including `.editorlm/tools/extract-source.py`, are product tools rather than user project artifacts. Do not edit them while working in a project. If a tool has a defect, explain it to the user and request an EditorLM product update; use the documented command only after the user approves the source scope.

User-created or user-customized skills belong under `.editorlm/skills/user/<skill-name>/SKILL.md`. Do not edit the default skills directly under `.editorlm/skills/defaults/` during project work; the user restores those defaults through the EditorLM panel. Keep custom workflow guidance in `.editorlm/skills/user/` instead.

1. **Project Subject:** After the user selects the one **Set Up EditorLM Project** button, begin when they say `Start Project Definition`. Read `.editorlm/skills/defaults/project-context/SKILL.md`, ask its focused opening question, and help create `PROJECT_SUBJECT.md`. Do not treat it as active until the user explicitly approves it.
2. **Sources and Basic Source Processing:** The user adds original material under `sources/` (subfolders are allowed). EditorLM automatically records only bounded local readiness work. Do not treat `.editorlm/excluded/` as evidence unless the user explicitly asks.
3. **Source processing:** When the user says `Assess Source Processing` or asks naturally about source processing, read `.editorlm/skills/defaults/source-processing/SKILL.md`. Assess the complete registered corpus by relevance, recommend the smallest useful scope, explain why, and wait for explicit user approval before reading sources or creating artifacts. For approved sources that need conversion, run the documented local extractor first; then use normal Codex reading and file-edit approval behavior to create the documented knowledge artifacts. Do not look for or invent a `process-sources` command. A user may edit the skill to waive approval. If processing was declined, raise it again only when it becomes materially useful.
4. **Embedding:** Only after summaries exist, assess whether semantic retrieval would materially help. When the user says `Assess Embedding`, read `.editorlm/skills/defaults/embedding/SKILL.md`, recommend the smallest useful set of already processed sources, and wait for separate approval before any embedding tool or provider call. Do not embed automatically. A changed Project Subject creates new context summaries only; it does not rebuild unchanged vectors.
5. **Evidence, planning, and writing:** Before proposing `PLAN.md`, identify likely missing material. Ask whether the user wants to add it, explicitly authorize a web search, or proceed without it. Repeat this rule during research, planning, drafting, review, and verification. Never search, download, or rely on external material without permission. Then use the normal Codex approval flow to create and revise visible Markdown artifacts at the project root.

## User-facing workflow language

Never expose EditorLM's internal implementation stages or stage numbers to the user. Describe the work in terms the user can act on: **source processing**, **local source conversion**, **visual description**, **summarization**, or **embedding**.

## Next-step rule

At the end of every response, state one clear next action the user can take. Never end with an answer that leaves the user without direction. Keep the direction brief and relevant: a focused question, an approval request, a phrase to type, or a file/action to review.
