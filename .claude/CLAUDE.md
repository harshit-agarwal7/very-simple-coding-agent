# Project Guidelines

## Structure
- Source code in `src/`, tests in `tests/`, config in `config/`
- One module = one responsibility

## Code Standards
- Type hints on all function signatures
- Google-style docstrings on all public functions
- Use `logging`, never `print()` for diagnostics
- No bare `except:` — always catch specific exceptions, log meaningful error messages.
- No hardcoded secrets or magic numbers
- No premature abstractions: don't create templates or helpers that are only used once. Inline values unless they're shared across 2+ call sites.

## Dependencies
- Justify any new dependency before adding it
- Pin versions in pyproject.toml
- Use virtual environments. Never install globally

## Testing
- Think about how you would verify the working of any code you add - first write the tests and then go about writing the code.
- Write tests alongside implementation, not after
- Run tests before presenting work as done
- Ensure that whenever you make a change, you check if any existing tests need to be updated.
- Run `ruff format`, `ruff check`, and `mypy` before finishing

## Others
- Use the astarl ecosystem (uv, ruff, etc) as much as possible for requirement management, linitng, etc
- Suggest a commit message when you are done with a set of changes. Always run `git diff HEAD` and `git status` to see the full picture of all uncommitted changes across the entire repo — not just what was changed in the current session.
