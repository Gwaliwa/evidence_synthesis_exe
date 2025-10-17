# Build & Test: Evidence Synthesis (.exe)

## Local Windows Build (recommended first)
1. Install Python 3.11 and Git.
2. PowerShell:
   ```ps1
   python -m venv .venv
   .venv\Scripts\Activate
   pip install --upgrade pip
   pip install -r windows-build-requirements.txt
   pip install pyinstaller
   choco install -y tesseract
   # Optional for pdf2image
   # choco install -y poppler
   pyinstaller --noconfirm app1.spec
   ```
3. Run:
   ```ps1
   .\dist\EvidenceSynthesisApp\EvidenceSynthesisApp.exe
   ```
   No sample PDFs are bundled. Use your own PDFs via the app upload. `dist\EvidenceSynthesisApp\samples\`.

## GitHub Actions (Windows)
- Commit `app1.py`, `app1.spec`, `windows-build-requirements.txt`, PDFs, and `.github/workflows/windows-exe.yml`.
- Push or trigger **Actions → Build Windows EXE**.
- Download the artifact `EvidenceSynthesisApp-windows` after success.

## Tips
- If runtime import errors occur, add `--hidden-import <module>` to spec or uncomment the `collect_submodules` lines.
- Start with `--onedir` (spec) before moving to `--onefile`.