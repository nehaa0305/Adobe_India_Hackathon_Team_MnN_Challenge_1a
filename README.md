# 📄 PDF Heading Extractor

A robust tool to extract hierarchical headings from PDF documents using visual layout analysis and font-based scoring. Powered by [LayoutParser](https://layout-parser.github.io/) and deep learning (Detectron2), it achieves precise and reliable heading detection.

---

## 🚀 Features

- 🔍 **LayoutParser Integration**: Uses a deep learning model (Detectron2) to detect structural elements.
- 📐 **Multi-Modal Analysis**: Combines layout detection and text feature scoring.
- 🧠 **Intelligent Scoring**:
  - Font size percentile-based detection (90th, 75th)
  - Layout-aware position scoring
  - Bold/Italic formatting awareness
  - Visual confirmation using LayoutParser
- 🧹 **Noise Filtering**:
  - Detects and removes non-headings (dates, URLs, page numbers)
  - Strong vs. weak pattern-based classification
- 📊 **Hierarchical Classification**: Headings grouped and ranked (H1/H2/H3) using font size clustering

---

## 📦 Requirements

Install packages via `pip`:

```bash
pip install PyMuPDF numpy layoutparser torch torchvision opencv-python pillow pandas
pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git"
