"""Run once: saves your Gemini key to .streamlit/secrets.toml so you never type it again."""
import getpass, pathlib

key = getpass.getpass("Paste your Gemini API key (typing is hidden), then press Enter: ").strip()
if not key:
    raise SystemExit("No key entered. Nothing was saved.")
model = input("Model name (press Enter for gemini-3.8-flash): ").strip() or "gemini-3.8-flash"
folder = pathlib.Path(__file__).parent / ".streamlit"
folder.mkdir(exist_ok=True)
(folder / "secrets.toml").write_text(f'GEMINI_API_KEY = "{key}"\nGEMINI_MODEL = "{model}"\n', encoding="utf-8")
print("\nSaved to .streamlit/secrets.toml. Now start the app with run_app.bat")
