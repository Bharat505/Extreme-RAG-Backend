import re
import uuid  # For unique chunk IDs
import os
import json
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ✅ Get backend root directory
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_pdfs")  # ✅ Ensure correct directory path

def cleanup_old_files(keep_last_n=5):
    """Keeps only the last `n` processed files in `processed_pdfs/`."""
    files = sorted(
        [os.path.join(PROCESSED_DIR, f) for f in os.listdir(PROCESSED_DIR) if f.endswith(".json")],
        key=os.path.getmtime,
    )

    # Keep only the last `n` files, delete the rest
    for file in files[:-keep_last_n]:
        os.remove(file)
        print(f"🗑 Deleted old file: {file}")

def split_sentences(text):
    """
    Splits text into sentences using regex.
    Handles '.', '?', and '!' as sentence boundaries.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_extracted_text_with_page_references(extracted_pdfs, base_word_threshold=3000):
    """
    Combines extracted text into chunks while keeping source page references.
    """
    all_chunks = []

    for pdf_id, extracted_data in extracted_pdfs.items():
        file_name = extracted_data["file_name"]
        total_words = sum(len(p.get('text', '').split()) for p in extracted_data['pages'] if p.get('text', '').strip())
        avg_words_per_page = total_words / max(len(extracted_data['pages']), 1)
        word_threshold = max(base_word_threshold, int(avg_words_per_page * 1.5))

        current_chunk_id = 1
        current_chunk_text = []
        current_page_range = set()
        current_word_count = 0
        current_tables = []
        current_images = []
        current_ocr_screenshots = []

        for page_info in extracted_data["pages"]:
            page_number = page_info.get("page_number", 0)
            page_text = page_info.get("text", "").strip()
            page_tables = [{"page": page_number, "table_data": table} for table in page_info.get("tables", [])]
            page_images = [{"page": page_number, "image_data": img} for img in page_info.get("embedded_images", [])]
            page_screenshot = {"page": page_number, "screenshot": page_info.get("page_image", None)}

            if not page_text and not page_tables and not page_images:
                continue

            sentences = split_sentences(page_text)
            processed_sentences = [f"[Page {page_number}] {s}" for s in sentences if s.strip()]
            words_in_page = sum(len(s.split()) for s in processed_sentences)

            if current_word_count + words_in_page > word_threshold:
                chunk_text_combined = " ".join(current_chunk_text)
                page_min, page_max = min(current_page_range), max(current_page_range)
                all_chunks.append({
                    "pdf_id": pdf_id,
                    "file_name": file_name,
                    "chunk_id": f"{pdf_id}_{current_chunk_id}",
                    "page_range": f"{page_min}-{page_max}" if page_min != page_max else f"{page_min}",
                    "chunk_text": chunk_text_combined,
                    "tables": current_tables,
                    "embedded_images": current_images,
                    "ocr_screenshots": current_ocr_screenshots
                })
                current_chunk_id += 1
                current_chunk_text = []
                current_page_range = set()
                current_word_count = 0
                current_tables = []
                current_images = []
                current_ocr_screenshots = []

            current_chunk_text.extend(processed_sentences)
            current_page_range.add(page_number)
            current_word_count += words_in_page
            current_tables.extend(page_tables)
            current_images.extend(page_images)
            if page_screenshot:
                current_ocr_screenshots.append(page_screenshot)

        if current_chunk_text or current_tables or current_images or current_ocr_screenshots:
            chunk_text_combined = " ".join(current_chunk_text)
            page_min, page_max = min(current_page_range), max(current_page_range)
            all_chunks.append({
                "pdf_id": pdf_id,
                "file_name": file_name,
                "chunk_id": f"{pdf_id}_{current_chunk_id}",
                "page_range": f"{page_min}-{page_max}" if page_min != page_max else f"{page_min}",
                "chunk_text": chunk_text_combined,
                "tables": current_tables,
                "embedded_images": current_images,
                "ocr_screenshots": current_ocr_screenshots
            })

    return all_chunks

def process_chunking(extracted_data=None):
    """Reads extracted PDFs and chunks the text."""
    cleanup_old_files(keep_last_n=5)
    if extracted_data is None:
        extracted_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".json")]
        extracted_data = {file: json.load(open(os.path.join(PROCESSED_DIR, file))) for file in extracted_files}
    
    chunked_results = chunk_extracted_text_with_page_references(extracted_data)
    
    if not isinstance(chunked_results, list):
        raise ValueError("❌ Error: process_chunking() must return a list of dictionaries")
    
    with open(os.path.join(PROCESSED_DIR, "chunked_data.json"), "w") as f:
        json.dump(chunked_results, f, indent=4)
    
    return chunked_results
