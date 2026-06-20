# Code style

- Prefer simple code over complicated code. Default to the smallest, most direct implementation that solves the problem in front of you.
- Before writing new code, check whether an existing function/method already does what you need. Don't duplicate logic that exists elsewhere.
- Don't use static methods unless there's no other option, or an instance method would be noticeably more complicated to use. Default to instance methods.
- Don't create dataclasses unless you ask first.
- After every code change update the .md docs in Docs/ folder.
- Read the needed Docs file before making any changes so you can see if there is a class or function that can be used and to understand the big picture.
- Before writing or running tests, read `Docs/Testing.md` — it maps which `Temp/tests/*.py` file covers which subsystem, the shared `conftest.py` fixtures, and the conventions to mirror when adding new tests. Don't add a new test file without checking whether an existing one already covers that subsystem.
