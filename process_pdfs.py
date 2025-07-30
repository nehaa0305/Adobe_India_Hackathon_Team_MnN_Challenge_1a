

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import fitz  # PyMuPDF
import numpy as np
from collections import defaultdict, Counter
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
import cv2
from PIL import Image
import layoutparser as lp
import torch
import pandas as pd
from dataclasses import dataclass
import io # Added for pdf_page_to_image


@dataclass
class HeadingCandidate:
    text: str
    page: int
    bbox: Tuple[float, float, float, float]
    font_size: float
    font_name: str
    is_bold: bool
    confidence: float
    layout_type: str
    hierarchy_level: int

class AdvancedPDFHeadingExtractor:
    def __init__(self):
        self.setup_logging()
        self.layout_model = None # Initialize layout_model to None
        self.setup_models()
        
        # Refined heading patterns with negative lookbehind/lookahead
        self.strong_heading_patterns = [
            r'^(?:chapter|section|part|unit|module)\s*\d+\s*[:\-\.]?\s*.+$',
            r'^\d+\.(?:\d+\.)*\s+[A-Z][^.]*$',  # Numbered sections
            r'^[IVX]+\.\s+[A-Z][^.]*$',  # Roman numerals
            r'^(?:introduction|conclusion|abstract|summary|methodology|results|discussion|references|bibliography|appendix)$',
        ]
        
        self.weak_heading_patterns = [
            r'^[A-Z][A-Z\s]{10,}$',  # All caps (longer)
            r'^\d+\.\s+.+$',  # Simple numbered
        ]
        
        # Strong negative patterns
        self.negative_patterns = [
            r'^\d+$',  # Pure numbers
            r'^page\s+\d+',  # Page numbers
            r'^figure\s+\d+',  # Figure captions
            r'^table\s+\d+',  # Table captions
            r'^fig\.\s*\d+',
            r'^tab\.\s*\d+',
            r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}',  # Dates
            r'^https?://',  # URLs
            r'^www\.',
            r'@\w+\.',  # Emails
            r'^[^\w\s]*$',  # Only symbols
            r'^\s*$',  # Empty
            r'^.{1,3}$',  # Too short
            r'^.{200,}$',  # Too long for headings
        ]
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def setup_models(self):
        """Initialize LayoutParser models"""
        try:
            # Use a lightweight but accurate model
            self.layout_model = lp.models.detectron2.Detectron2LayoutModel(
                config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                enforce_cpu=True  # for Colab/local
            )
            self.logger.info("LayoutParser model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load LayoutParser model: {e}")
            self.layout_model = None
    
    def pdf_page_to_image(self, page: fitz.Page, dpi: int = 150) -> np.ndarray:
        """Convert PDF page to image for layout analysis"""
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("ppm")
        
        # Convert to PIL Image then to numpy array
        img = Image.open(io.BytesIO(img_data))
        return np.array(img)
    
    def extract_layout_information(self, doc: fitz.Document) -> Dict[int, List]:
        """Extract layout information using LayoutParser"""
        layout_info = {}
        
        if self.layout_model is None:
            self.logger.warning("LayoutParser model not available, using fallback")
            return layout_info
        
        for page_num in range(min(len(doc), 10)):  # Limit to first 10 pages for speed
            try:
                page = doc[page_num]
                img = self.pdf_page_to_image(page)
                
                # Run layout detection
                layout = self.layout_model.detect(img)
                
                # Convert layout blocks to our format
                page_layouts = []
                for block in layout:
                    if block.type in ['Title', 'Text']:
                        page_layouts.append({
                            'type': block.type,
                            'bbox': (block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2),
                            'confidence': block.score
                        })
                
                layout_info[page_num] = page_layouts
                
            except Exception as e:
                self.logger.warning(f"Layout detection failed for page {page_num}: {e}")
                layout_info[page_num] = []
        
        return layout_info
    
    def extract_text_with_metadata(self, doc: fitz.Document) -> List[Dict]:
        """Extract text with comprehensive metadata"""
        text_blocks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get text dictionary with detailed formatting
            text_dict = page.get_text("dict")
            page_height = page.rect.height
            page_width = page.rect.width
            
            for block in text_dict["blocks"]:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    line_text = ""
                    font_sizes = []
                    font_names = []
                    flags = []
                    
                    for span in line["spans"]:
                        line_text += span["text"]
                        font_sizes.append(span["size"])
                        font_names.append(span["font"])
                        flags.append(span["flags"])
                    
                    if not line_text.strip():
                        continue
                    
                    # Calculate font properties
                    avg_font_size = np.mean(font_sizes) if font_sizes else 12
                    dominant_font = Counter(font_names).most_common(1)[0][0] if font_names else ""
                    
                    # Determine formatting
                    is_bold = any(flag & 2**4 for flag in flags)  # Bold flag
                    is_italic = any(flag & 2**1 for flag in flags)  # Italic flag
                    
                    # Get bounding box
                    bbox = line["bbox"]
                    
                    # Calculate position metrics
                    distance_from_top = bbox[1] / page_height
                    distance_from_left = bbox[0] / page_width
                    text_width = (bbox[2] - bbox[0]) / page_width
                    
                    text_blocks.append({
                        'text': line_text.strip(),
                        'page': page_num + 1,
                        'bbox': bbox,
                        'font_size': avg_font_size,
                        'font_name': dominant_font,
                        'is_bold': is_bold,
                        'is_italic': is_italic,
                        'distance_from_top': distance_from_top,
                        'distance_from_left': distance_from_left,
                        'text_width': text_width,
                        'char_count': len(line_text.strip()),
                        'word_count': len(line_text.strip().split())
                    })
        
        return text_blocks
    
    def calculate_document_stats(self, text_blocks: List[Dict]) -> Dict:
        """Calculate document-wide statistics"""
        if not text_blocks:
            return {}
        
        font_sizes = [block['font_size'] for block in text_blocks]
        char_counts = [block['char_count'] for block in text_blocks]
        
        return {
            'avg_font_size': np.mean(font_sizes),
            'std_font_size': np.std(font_sizes),
            'max_font_size': max(font_sizes),
            'min_font_size': min(font_sizes),
            'p75_font_size': np.percentile(font_sizes, 75),
            'p90_font_size': np.percentile(font_sizes, 90),
            'avg_char_count': np.mean(char_counts),
            'median_char_count': np.median(char_counts)
        }
    
    def is_negative_match(self, text: str) -> bool:
        """Check if text matches negative patterns"""
        text_clean = text.strip().lower()
        for pattern in self.negative_patterns:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return True
        return False
    
    def calculate_heading_score(self, block: Dict, doc_stats: Dict, layout_info: Dict) -> float:
        """Calculate comprehensive heading score"""
        if self.is_negative_match(block['text']):
            return 0.0
        
        score = 0.0
        
        # Font size analysis (more sophisticated)
        if block['font_size'] > doc_stats['p90_font_size']:
            score += 0.4
        elif block['font_size'] > doc_stats['p75_font_size']:
            score += 0.25
        elif block['font_size'] > doc_stats['avg_font_size'] * 1.1:
            score += 0.15
        
        # Formatting weight
        if block['is_bold']:
            score += 0.25
        if block['is_italic']:
            score += 0.1
        
        # Position analysis
        if block['distance_from_top'] < 0.1:  # Top of page
            score += 0.2
        if block['distance_from_left'] < 0.1:  # Left aligned
            score += 0.15
        
        # Length analysis (refined)
        word_count = block['word_count']
        char_count = block['char_count']
        
        if 2 <= word_count <= 8:  # Optimal heading length
            score += 0.2
        elif 9 <= word_count <= 15:
            score += 0.1
        elif word_count > 25:  # Too long for heading
            score -= 0.3
        
        if char_count < 5:  # Too short
            score -= 0.2
        
        # Pattern matching (strong patterns)
        text_lower = block['text'].lower()
        for pattern in self.strong_heading_patterns:
            if re.match(pattern, text_lower, re.IGNORECASE):
                score += 0.35
                break
        else:
            # Check weak patterns
            for pattern in self.weak_heading_patterns:
                if re.match(pattern, block['text'], re.IGNORECASE):
                    score += 0.15
                    break
        
        # Layout information bonus
        page_layout = layout_info.get(block['page'] - 1, [])
        for layout_block in page_layout:
            if layout_block['type'] == 'Title':
                # Check if our text block overlaps with detected title
                if self.bbox_overlap(block['bbox'], layout_block['bbox']) > 0.5:
                    score += 0.3
        
        # Structural indicators
        text = block['text']
        if re.match(r'^\d+\.', text):  # Numbered
            score += 0.2
        if re.match(r'^[A-Z][A-Z\s]+$', text) and len(text) > 5:  # All caps
            score += 0.15
        if text.endswith(':'):  # Ends with colon
            score += 0.15
        
        # Capitalization analysis
        words = text.split()
        if len(words) > 1:
            title_case_words = sum(1 for word in words if word and word[0].isupper())
            if title_case_words / len(words) >= 0.7:  # Mostly title case
                score += 0.1
        
        return max(0.0, score)
    
    def bbox_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        
        return intersection_area / bbox1_area if bbox1_area > 0 else 0.0
    
    def assign_hierarchy_levels(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Assign H1, H2, H3 levels based on font size and context"""
        if not candidates:
            return candidates
        
        # Sort by font size (descending) and then by confidence
        candidates.sort(key=lambda x: (-x.font_size, -x.confidence))
        
        # Group by similar font sizes
        font_groups = []
        current_group = [candidates[0]]
        
        for i in range(1, len(candidates)):
            if abs(candidates[i].font_size - current_group[0].font_size) <= 1.0:
                current_group.append(candidates[i])
            else:
                font_groups.append(current_group)
                current_group = [candidates[i]]
        
        if current_group:
            font_groups.append(current_group)
        
        # Assign levels
        for i, group in enumerate(font_groups):
            # Limit to actual hierarchy levels you want to define (e.g., H1, H2, H3)
            if i < 3: 
                for candidate in group:
                    candidate.hierarchy_level = i + 1
            else:
                # Assign to the lowest defined level if more groups exist
                for candidate in group:
                    candidate.hierarchy_level = 3 
        
        # Sort by page and position for final output
        candidates.sort(key=lambda x: (x.page, x.bbox[1]))
        
        return candidates
    
    def extract_headings_from_pdf(self, pdf_path: str) -> Dict:
        """Main extraction function"""
        try:
            start_time = time.time()
            
            with fitz.open(pdf_path) as doc:
                # Extract text with metadata
                text_blocks = self.extract_text_with_metadata(doc)
                
                if not text_blocks:
                    return {
                        "title": Path(pdf_path).stem,
                        "outline": [],
                        "processing_time": time.time() - start_time,
                        "error": "No text found"
                    }
                
                # Calculate document statistics
                doc_stats = self.calculate_document_stats(text_blocks)
                
                # Extract layout information
                layout_info = self.extract_layout_information(doc)
                
                # Score all text blocks
                candidates = []
                for block in text_blocks:
                    score = self.calculate_heading_score(block, doc_stats, layout_info)
                    
                    # Use higher threshold for better precision
                    if score >= 0.5:
                        candidate = HeadingCandidate(
                            text=block['text'],
                            page=block['page'],
                            bbox=block['bbox'],
                            font_size=block['font_size'],
                            font_name=block['font_name'],
                            is_bold=block['is_bold'],
                            confidence=score,
                            layout_type="detected",
                            hierarchy_level=0
                        )
                        candidates.append(candidate)
                
                # If no high-confidence candidates, lower threshold slightly
                if not candidates:
                    for block in text_blocks:
                        score = self.calculate_heading_score(block, doc_stats, layout_info)
                        if score >= 0.35:
                            candidate = HeadingCandidate(
                                text=block['text'],
                                page=block['page'],
                                bbox=block['bbox'],
                                font_size=block['font_size'],
                                font_name=block['font_name'],
                                is_bold=block['is_bold'],
                                confidence=score,
                                layout_type="fallback",
                                hierarchy_level=0
                            )
                            candidates.append(candidate)
                
                # Assign hierarchy levels
                candidates = self.assign_hierarchy_levels(candidates)
                
                # Get document title
                title = self.extract_title(doc, text_blocks) or Path(pdf_path).stem
                
                # Format output
                outline = []
                level_map = {1: "H1", 2: "H2", 3: "H3"}
                
                for candidate in candidates:
                    outline.append({
                        "level": level_map.get(candidate.hierarchy_level, "H3"),
                        "text": candidate.text,
                        "page": candidate.page
                    })
                
                processing_time = time.time() - start_time
                
                return {
                    "title": title,
                    "outline": outline,
                    "processing_time": processing_time,
                    "headings_found": len(outline)
                }
                
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {
                "title": Path(pdf_path).stem,
                "outline": [],
                "processing_time": 0,
                "error": str(e)
            }
    
    def extract_title(self, doc: fitz.Document, text_blocks: List[Dict]) -> Optional[str]:
        """Extract document title"""
        # Try metadata first
        metadata = doc.metadata
        if metadata.get("title") and len(metadata["title"].strip()) > 3:
            return metadata["title"].strip()
        
        # Look for title on first page
        first_page_blocks = [b for b in text_blocks if b['page'] == 1]
        if not first_page_blocks:
            return None
        
        # Sort by font size and position
        first_page_blocks.sort(key=lambda x: (-x['font_size'], x['distance_from_top']))
        
        # Look for the largest, well-positioned text that looks like a title
        for block in first_page_blocks[:3]:
            text = block['text'].strip()
            
            if (len(text) > 10 and 
                len(text) < 150 and 
                not self.is_negative_match(text) and
                block['font_size'] >= np.mean([b['font_size'] for b in first_page_blocks]) * 1.1):
                return text
        
        return None
    
    def process_directory(self, input_dir: str, output_dir: str, max_workers: int = 2):
        """Process all PDFs in directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(input_path.glob("*.pdf"))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {input_dir}")
            return
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        processed = 0
        failed = 0
        
        # Use fewer workers to avoid memory issues with deep learning models
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.extract_headings_from_pdf, str(pdf_file)): pdf_file 
                for pdf_file in pdf_files
            }
            
            for future in as_completed(future_to_file):
                pdf_file = future_to_file[future]
                try:
                    result = future.result()
                    
                    # Clean result for JSON output
                    clean_result = {
                        "title": result["title"],
                        "outline": result["outline"]
                    }
                    
                    # Save to JSON
                    output_file = output_path / f"{pdf_file.stem}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(clean_result, f, indent=2, ensure_ascii=False)
                    
                    processed += 1
                    self.logger.info(f"✓ {pdf_file.name}: {len(result['outline'])} headings "
                                   f"({result.get('processing_time', 0):.1f}s)")
                    
                except Exception as e:
                    failed += 1
                    self.logger.error(f"✗ {pdf_file.name}: {str(e)}")
        
        self.logger.info(f"Complete: {processed} success, {failed} failed")

def main():
    """Main function"""
    
    # Get input and output directories from user
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Validate input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ask for number of workers (default 2 for deep learning models)
    workers_input = "2"
    workers = 2
    try:
        workers_input = input("Enter number of worker threads (default: 2): ").strip()
        if workers_input.isdigit():
            workers = int(workers_input)
    except ValueError:
        print("Invalid input for workers, using default of 2.")

    
    print(f"\nProcessing PDFs from: {input_dir}")
    print(f"Output will be saved to: {output_dir}")
    print(f"Using {workers} worker threads")
    print("Loading deep learning models...\n")
    
    extractor = AdvancedPDFHeadingExtractor()
    extractor.process_directory(input_dir, output_dir, workers)

if __name__ == "__main__":
    main()

