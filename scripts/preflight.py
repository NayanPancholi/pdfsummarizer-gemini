"""Run every credential check before commits and before the final push."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_gemini import validate as validate_gemini  # noqa: E402
from validate_github import validate as validate_github  # noqa: E402


def main() -> int:
    print("=== Preflight checks ===")
    gemini_ok = validate_gemini()
    github_ok = validate_github()

    print("=== Summary ===")
    print(f"Gemini: {'PASS' if gemini_ok else 'FAIL'}")
    print(f"GitHub: {'PASS' if github_ok else 'FAIL'}")
    if gemini_ok and github_ok:
        print("All preflight checks passed.")
        return 0
    print("Preflight failed. Fix the issues above and retry.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
