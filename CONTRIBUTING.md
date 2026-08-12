# Contributing to Nina

Thank you for contributing to Nina! To maintain high software architecture quality, clean code standards, and reproducibility, please follow these engineering guidelines.

---

## 1. Code Style & Quality Standards

Nina strictly enforces clean code standards, explicit typing, and modular architecture.

### 1.1 Python Code Standards
* **Python Version:** Python 3.10+
* **Formatting:** Code MUST be formatted with `black` (line length 100).
* **Linting:** Enforce strict checks with `ruff`.
* **Type Hints:** All function parameters and return types MUST have explicit type annotations checked by `mypy`.
* **Docstrings:** Use Google-style docstrings for all public modules, classes, interfaces, and methods.

---

## 2. Environment Setup

```bash
# Clone repository and navigate to Nina root
cd Nina

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies in editable mode with dev extras
pip install -e .[dev,eval]
```

---

## 3. Running Quality Checks

Before committing code or opening a pull request, run the static analysis suite:

```bash
# 1. Format check
black --check src tests

# 2. Lint check
ruff check src tests

# 3. Type check
mypy src

# 4. Unit & Integration test suite
pytest
```

---

## 4. Git Branching & Commit Conventions

### 4.1 Branch Naming Strategy
* `main`: Production-ready releases.
* `develop`: Active integration branch.
* `feature/<feature-name>`: New capabilities or layers (e.g., `feature/faster-whisper-stt`).
* `fix/<bug-name>`: Bug fixes (e.g., `fix/audio-buffer-overflow`).

### 4.2 Commit Messages
Follow Conventional Commits format:
```
<type>(<scope>): <short summary>

[optional body]
```
* **Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.
* **Example:** `feat(emotion): implement intensity calculator heuristic engine`

---

## 5. Pull Request Guidelines

1. Ensure all tests (`pytest`) pass cleanly without errors or warnings.
2. Include unit test coverage for any new interface or logic component.
3. Update relevant documentation (`ARCHITECTURE.md`, `MODEL_SELECTION.md`, `README.md`) if architecture or configuration contracts change.
4. Obtain code review approval from the Lead Software Architect.
