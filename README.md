# Adobe_India_Hackathon_Team_MnN_Challenge_1a
#  LlamaParse-Based Multilingual PDF Outline Extractor

This project parses **PDF documents** and generates a clean, structured **outline JSON** file. It uses LlamaParse to extract hierarchical headings (`H1`, `H2`, `H3`) from documents including multilingual PDFs using OCR preprocessing if needed.

---

##  Features

-  **Multilingual PDF Handling** (Hindi, Tamil, French, etc.)
- Supports scanned PDFs via **OCR (Tesseract)**
-  **Semantic Parsing** using LlamaParse with layout awareness
- Fast Execution: **Processes 50-page PDFs in under 10 seconds**
-  JSON output in competition-specific format
- Deduplicates and sorts headings by correct page number
-  Batch processing for all PDFs in a given folder

---



## ⚙️ System Requirements

- **Python 3.9+**
- **Tesseract OCR installed**
- **Poppler installed** (for `pdf2image`)
- Internet access (for LlamaParse API)

> 💡 Supports up to **16GB RAM** and **8 CPU cores**.

---

##  Models Used

### 1. **LlamaParse**
- Uses `llama-parse` cloud API with customized `parsing_instruction` to semantically extract headings.
- Understands layout semantics: font size, position, boldness — but avoids false positives (e.g., table of contents lines).

### 2. **OCR + Language Detection**
- `pdf2image` to convert PDFs to images
- `pytesseract` for OCR
- `langdetect` or `Argos` to auto-detect language
- OCR language is mapped via `lang_map` to appropriate `tesseract` model

---

## 🧪 Testing Strategy

| Test Type        | Details                                                  |
|------------------|-----------------------------------------------------------|
| Simple PDFs    | Basic single-column documents with clear headings         |
| Complex PDFs   | Multi-column layouts, embedded tables/images, bold texts  |
|  Large PDFs     | 50+ pages processed in under 10 seconds                   |
| Multilingual   | PDFs in Hindi, French, Tamil — verified headings are preserved correctly |

---

## ⚡️ Performance Optimizations

| Constraint           | Optimization Applied                                       |
|----------------------|------------------------------------------------------------|
|  <10s Total Time    | ✓ Async LlamaParse batching, minimal markdown IO          |
|  <16GB RAM         | ✓ Stream OCR image processing page-by-page                |
|  Efficient CPU     | ✓ Multiprocessing-safe queues for language detection + OCR |
|  Memory Cleanup    | ✓ Use `tempfile` for OCR intermediate files                |

---

## Clustering (Optional Extension)

This pipeline is compatible with post-processing **clustering** of extracted headings using semantic models (like `SBERT`) to:

- Group similar section headings across document types
- Automatically identify common structural patterns (e.g., “Overview” + “Introduction”)

This can be added as a downstream analytics task.

---




## Docker Instructions

You can run the entire LlamaParse-based multilingual PDF processing pipeline using Docker:

docker build -t llama-notebook .
docker run --rm llama-notebook




