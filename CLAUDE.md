# Code style

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
- Read the needed Docs file before making any changes so you can see if there is a class or function that can be used and to understand the big picture.
- Before writing or running tests, read `Docs/Testing.md` — it maps which `Temp/tests/*.py` file covers which subsystem, the shared `conftest.py` fixtures, and the conventions to mirror when adding new tests. Don't add a new test file without checking whether an existing one already covers that subsystem.
- After every code or doc change, add a dated entry to `Docs/ChangeLog.md` (newest entry on top, see its own "Convention for new entries" section) summarizing what changed and why, with the files touched. This is how another agent or session — possibly working in parallel — finds out what happened since it last looked, without re-reading every diff.
