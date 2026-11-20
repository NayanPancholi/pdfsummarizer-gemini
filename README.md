# PDF Summarizer (LangChain + Gemini)

A Streamlit web app that extracts text from a PDF with `pypdf`, splits it into
token-aware chunks, and summarizes it with a LangChain LCEL map-reduce chain
backed by Google Gemini.

## Prerequisites

- Python 3.13
- Git for Windows
- GitHub CLI (`gh`)
- A Gemini API key

Install the Windows tooling with:

```powershell
winget install --id Git.Git --scope user -e --accept-source-agreements --accept-package-agreements
winget install --id GitHub.cli --scope user -e --accept-source-agreements --accept-package-agreements
```

Restart your terminal after installation so `git` and `gh` are on `PATH`.

## Setup

```powershell
# From the project directory
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
copy .env.example .env
```

Edit `.env` and set your real values:

```dotenv
GEMINI_API_KEY=your_real_api_key
GEMINI_MODEL=gemini-3.1-flash-lite
```

Optional chunking settings: `PDF_CHUNK_SIZE` (default `4000`) and
`PDF_CHUNK_OVERLAP` (default `400`).

`.env` is listed in `.gitignore` and must never be committed.

## Validate credentials

```powershell
.\.venv\Scripts\python.exe scripts\validate_gemini.py
.\.venv\Scripts\python.exe scripts\validate_github.py
.\.venv\Scripts\python.exe scripts\preflight.py
```

If GitHub reports that you are not logged in:

```powershell
gh auth login
```

Choose **GitHub.com → HTTPS → Login with a web browser** and complete the
browser flow. Re-run the preflight afterwards.

## Run the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL shown in the terminal, upload a PDF, pick a summary style,
and click **Summarize**. Styles: `concise`, `detailed`, `bullets`, `insights`.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Project structure

```text
app.py                    Streamlit UI
src/config.py             Environment configuration
src/pdf_loader.py         pypdf text extraction
src/chunker.py            Token-aware splitting
src/summarizer.py         LCEL map-reduce summary chains
scripts/validate_gemini.py  Live Gemini API key check
scripts/validate_github.py  GitHub login check
scripts/preflight.py      Runs both checks
tests/                    Pytest suite (mocked Gemini)
```

## Publishing

The final phase creates the public repository and pushes the code:

```powershell
.\.venv\Scripts\python.exe scripts\preflight.py
git init
git add .
git commit -m "Initial PDF summarizer implementation"
gh repo create NayanPancholi/pdfsummarizer-gemini --public --source=. --remote=origin --push
```
