"""Validate GitHub CLI installation, login, and repository access."""

from __future__ import annotations

import shutil
import subprocess

TARGET_REPO = "NayanPancholi/pdfsummarizer-gemini"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def validate() -> bool:
    """Return True when GitHub CLI is installed and authenticated."""
    print("[GitHub] Checking GitHub CLI and login...")
    if shutil.which("gh") is None:
        print(
            "[GitHub] FAIL: 'gh' was not found on PATH. "
            "Install it with: winget install --id GitHub.cli --scope user"
        )
        return False

    status = _run(["gh", "auth", "status"])
    if status.returncode != 0:
        print("[GitHub] FAIL: not logged in. Run 'gh auth login' and retry.")
        return False

    user = _run(["gh", "api", "user", "--jq", ".login"])
    if user.returncode != 0:
        detail = (user.stderr or user.stdout).strip()
        print(f"[GitHub] FAIL: could not read the authenticated user - {detail}")
        return False
    print(f"[GitHub] Logged in as: {user.stdout.strip()}")

    repo = _run(["gh", "api", f"repos/{TARGET_REPO}", "--jq", ".full_name"])
    if repo.returncode == 0:
        print(f"[GitHub] Repository available: {repo.stdout.strip()}")
    elif "404" in (repo.stderr + repo.stdout):
        print(f"[GitHub] Repository {TARGET_REPO} does not exist yet (created in Phase 5).")
    else:
        detail = (repo.stderr or repo.stdout).strip()
        print(f"[GitHub] FAIL: could not check repository access - {detail}")
        return False

    print("[GitHub] PASS: GitHub CLI is installed and authenticated.")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if validate() else 1)
