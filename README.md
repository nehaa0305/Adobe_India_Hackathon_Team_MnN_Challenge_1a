📄 PDF Heading Extractor
A robust tool to extract hierarchical headings from PDF documents using visual layout analysis and font-based scoring. Powered by LayoutParser and deep learning (Detectron2), it achieves precise and reliable heading detection.

🚀 Features
🔍 LayoutParser Integration: Uses a deep learning model (Detectron2) to detect structural elements.

📐 Multi-Modal Analysis: Combines layout detection and text feature scoring.

🧠 Intelligent Scoring:

Font size percentile-based detection (90th, 75th)

Layout-aware position scoring

Bold/Italic formatting awareness

Visual confirmation using LayoutParser

🧹 Noise Filtering:

Detects and removes non-headings (dates, URLs, page numbers)

Strong vs. weak pattern-based classification

📊 Hierarchical Classification: Headings grouped and ranked (H1/H2/H3) using font size clustering.

📦 Requirements
Install packages via pip:

bash
Copy
Edit
pip install PyMuPDF numpy layoutparser torch torchvision opencv-python pillow pandas
pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git"
Or use the included requirements.txt:

text
Copy
Edit
PyMuPDF>=1.23.0  
numpy>=1.21.0  
layoutparser[layoutmodels]>=0.3.4  
torch>=1.9.0  
torchvision>=0.10.0  
opencv-python>=4.5.0  
Pillow>=8.0.0  
pandas>=1.3.0  
detectron2
📥 Note: The first run downloads a 150MB LayoutParser model. Ensure a stable internet connection.

🛠️ Usage
🔧 Command-Line Interface
bash
Copy
Edit
python pdf_heading_extractor.py
You’ll be prompted to enter:

📁 Input Directory (PDF files)

📂 Output Directory (for extracted JSON headings)

🧵 Number of worker threads (default: 4)

🐍 Python API
python
Copy
Edit
from pdf_heading_extractor import PDFHeadingExtractor

extractor = PDFHeadingExtractor()
result = extractor.extract_headings_from_pdf("document.pdf")
print(result)
⚙️ How It Works
1. 📸 LayoutParser Analysis
Converts PDF pages to images

Uses a Detectron2 model to identify layout blocks (e.g., Titles)

Tags Title blocks with high visual confidence

2. 🧾 Text Extraction with Metadata
Extracts text with positioning, font size, bold/italic metadata

Computes location-based statistics

3. 📊 Statistical Font Analysis
Computes 75th and 90th percentiles of font sizes

Robust against formatting outliers

Scores text based on percentile ranking

4. 🧮 Heading Scoring System
Feature	Weight
Font Size	40%
Layout Confirmation	30%
Formatting (Bold/Italic)	25%
Pattern Matching	35%
Position Analysis	20%
Negative Filtering	Aggressive

5. 🔤 Pattern Matching
✅ Strong patterns: "Chapter", numbered sections (1.1, 2.3.4)

⚠️ Weak patterns: ALL CAPS, single words

❌ Negative patterns: Page numbers, URLs, dates, emails

6. 🏷️ Hierarchical Classification
Groups by font size

Assigns heading levels: H1, H2, H3, etc.

Preserves order of appearance in the document

📁 Output Format
Outputs a JSON file for each PDF containing extracted headings with:

json
Copy
Edit
[
  {
    "text": "Introduction",
    "level": "H1",
    "page": 1,
    "bbox": [x0, y0, x1, y1],
    "font_size": 16.2
  },
  ...
]
📎 Notes
Tested on multi-column and academic-style PDFs.

For best results, ensure PDFs have embedded fonts and standard formatting.
