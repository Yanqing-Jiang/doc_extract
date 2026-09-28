<h1 align="center">Doc Extract</h1>

<p align="center"><b>Upload PDFs, describe the fields you want, and review what an LLM extracts in an editable table.</b><br>
A small August 2023 Streamlit prototype. It renders each page to an image, reads the text with Tesseract OCR, and asks GPT-3.5 to return the fields as JSON.</p>

<p align="center">
<a href="https://yanqing.app/project/llm-invoice-processor/"><b>Read the related invoice-processing case study</b></a> ·
<a href="#how-it-works">How it works</a> ·
<a href="#run-it-locally">Run it locally</a> ·
<a href="#legacy-constraints">Legacy constraints</a>
</p>

---

## How this relates to the case study

The yanqing.app case study covers a broader LLM invoice-processing workflow. This repository is not a deployment of that workflow. It's an earlier, standalone OCR-and-extraction prototype. It covers PDF extraction and table review; it does not implement ledger matching or an application-defined CSV export.

## How it works

| Step | Code | What happens |
|---|---|---|
| Upload | `st.file_uploader` | Accepts several PDFs and writes each one to a temporary file in the working directory |
| Render | `convert_pdf_to_images` | pypdfium2 renders every page at 300 DPI as a JPEG |
| OCR | `extract_text_from_img` | pytesseract reads each page, and the page texts are joined together |
| Extract | `extract_structured_data` | A LangChain `LLMChain` on `gpt-3.5-turbo-16k-0613` (temperature 0) gets the full text plus your field descriptions and is asked for a JSON array |
| Review | `st.data_editor` | The JSON from all files is combined into a pandas DataFrame that you can edit on the page |

### Data points

The text area starts with four fields written for one document layout: `WIC Number`, `Performance From`, `Performance To` and `Scan amount`. Edit it to describe your own fields. Its contents go into the prompt as plain text, so they don't need to be valid JSON.

## Run it locally

The commands below show the entrypoint. Resolve the PDF API, dependency and model compatibility issues in [Legacy constraints](#legacy-constraints) before using the app.

The entrypoint is `app.py`. You need the Tesseract binary installed on your system. `packages.txt` lists `tesseract-ocr`, which is the apt package list Streamlit Community Cloud reads.

```bash
# macOS: brew install tesseract
# Debian/Ubuntu: sudo apt-get install tesseract-ocr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pandas requests Pillow
export OPENAI_API_KEY="<your-openai-key>"
streamlit run app.py
```

Supply `OPENAI_API_KEY` to LangChain through the environment, as shown above.

## Legacy constraints

| Area | What to expect |
|---|---|
| Dependencies | `requirements.txt` has no version pins. It also leaves out `pandas`, `requests` and `Pillow`, which the code imports directly. |
| PDF rendering | The code calls `PdfDocument.render(pdfium.PdfBitmap.to_pil, ...)`, a version-sensitive multi-page API. Use a compatible pypdfium2 release or adapt the rendering code; this repo does not pin a version. |
| LangChain and model | `langchain.chat_models` and `LLMChain` are 2023-era imports. `gpt-3.5-turbo-16k-0613` is a legacy snapshot. Change `model=` to a model your account can use. |
| JSON parsing | The raw model output goes straight to `json.loads`. If the model adds prose or code fences, the app raises an error. |
| Reruns | Every widget interaction reruns the script. Uploaded files stay in place, so each rerun repeats OCR and the model calls for every file. |
| Temp files | Uploads are saved as `.csv`-suffixed temp files in the working directory and read back by path without `flush()`. If pdfium can't open a file, check this first. |
| Results | The app implements no export or download control, and it does not persist table edits. |
| OCR | Tesseract runs with its default settings and no image preprocessing. Scan quality limits the results. |

## Data handling

The full OCR text of each document goes to OpenAI and is also printed to the server console (`print(content)`).

## Files

| File | Purpose |
|---|---|
| [`app.py`](app.py) | PDF rendering, OCR, LLM extraction and the Streamlit UI |
| [`requirements.txt`](requirements.txt) | Unpinned Python dependencies (incomplete, see above) |
| [`packages.txt`](packages.txt) | System package list for Streamlit Community Cloud (`tesseract-ocr`) |

The repository has no license file.
