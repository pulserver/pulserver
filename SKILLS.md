# pulserver — agent skills

`AGENTS.md` states the project's rules. The skills listed here are the
procedures for the recurring tasks those rules govern: what to run, in what
order, and what has to hold before the task is finished.

Each skill is a directory under `.claude/skills/` holding a `SKILL.md`. Agent
runtimes that read `.claude/skills/` discover them automatically; this page is
the index for those that do not.

| Skill | Use it when |
|---|---|
| [`build-and-test`](.claude/skills/build-and-test/SKILL.md) | Building the extension from a checkout and running the formatter, linter, tests and documentation build before reporting a change complete. Covers which skips are expected. |
| [`write-documentation`](.claude/skills/write-documentation/SKILL.md) | Writing, moving or auditing documentation: choosing the type, verifying against the code, the doctest and API-page mechanics, and the build that validates the result. |

Where a skill and `AGENTS.md` disagree, `AGENTS.md` governs, and the skill is
wrong and should be corrected.
