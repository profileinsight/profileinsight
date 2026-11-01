# Instructions for AI agents

These rules apply to every AI agent working in this repository. They override any default behavior of the agent or its tooling.

## 1. Identity and attribution

- Never change the git author or committer name or email in any way. This covers global, system, and repository config, environment variables, and command-line flags.
- Never mention an AI agent, tool, or model in commit messages, trailers, co-author lines, contributor lists, documentation, code, or any other file. This includes Claude Code, Codex, and any specific model name.

## 2. Commits and pushes

- Never commit or push unless the user explicitly asks for it in the current request.
- Without such a request, leave all changes uncommitted, however much work was done.
- Before any commit, show the user the proposed commit message and the list of changed files. Ask the user to read and understand the changes themselves before approving.

## 3. No fallbacks

- All imports go at the top of a file. Never import inside functions or conditional branches, and never wrap an import in try/except.
- Code must not switch to a fallback implementation when the environment lacks something. A fallback hides the problem instead of solving it; fail loudly instead.

## 4. Minimal code

- Keep every file as short and simple as possible so that a human can read and understand it.
- Do not build complex engineering without explicit user approval. When a task seems to need it, stop, describe simpler alternatives, and let the user decide.

## 5. No traces of history

- Code, documents, and every other file must read as if they had always been this way.
- Do not write comments or text such as "previously...", "changed to...", "now uses...", "fixed...", or "updated...". Describe what is, not what was.

## 6. File headers

- Every code file starts with a header comment, above everything else, that carefully introduces the file: its purpose and the design idea behind it.
- The header covers every public component of the file: every publicly visible function, class, and method, each with its interface, the principles it follows, and an example of how to use it. None may be left out.
- Keep the header in step with the file: whenever the file changes, update its header.

## 7. Language

- Everything in this repository must be in English only: code, comments, documents, file names, data labels, and figures. No Chinese or any other language.
