import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import pandas as pd
import re
import io
import os
import base64
import uuid  # For generating unique PDF IDs
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ✅ Get backend root directory
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_pdfs")  # ✅ Ensure correct directory path

os.makedirs(PROCESSED_DIR, exist_ok=True)  # ✅ Ensure folder exists

def extract_full_content_structured(pdf_paths: list, output_dir=PROCESSED_DIR):
    """
    Extracts text, tables, and images from multiple PDFs using PyMuPDF + Tesseract OCR fallback.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_pdfs_data = {}

    for pdf_path in pdf_paths:
        doc = fitz.open(pdf_path)
        pdf_id = str(uuid.uuid4())
        file_name = os.path.basename(pdf_path)
        save_path = os.path.join(output_dir, f"{pdf_id}.json")

        structured_output = {
            "pdf_id": pdf_id,
            "file_name": file_name,
            "page_count": len(doc),
            "pages": []
        }

        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            page_data = {
                "page_number": page_number,
                "text": "",
                "tables": [],
                "embedded_images": [],
                "page_image": None,
                "metadata": {}
            }

            # Extract text
            page_text = page.get_text("text").strip()
            if page_text:
                page_data["text"] = clean_text(page_text)
            else:
                try:
                    zoom_matrix = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=zoom_matrix)
                    pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    pil_img = pil_img.convert("L")
                    custom_config = r"--psm 3 --oem 3"
                    ocr_text = pytesseract.image_to_string(pil_img, config=custom_config)
                    page_data["text"] = clean_text(ocr_text)
                except Exception as e:
                    print(f"❌ OCR failed on Page {page_number} in {file_name}: {e}")

            # Extract tables (Placeholder, actual extraction logic needed)
            page_data["tables"] = []

            # Extract embedded images
            try:
                img_list = page.get_images(full=True)
                embedded_imgs = []
                for img_info in img_list:
                    xref = img_info[0]
                    base_img = doc.extract_image(xref)
                    img_bytes = base_img["image"]
                    img_ext = base_img.get("ext", "png")
                    base64_img = base64.b64encode(img_bytes).decode("utf-8")
                    embedded_imgs.append({"extension": img_ext, "data": base64_img})
                page_data["embedded_images"] = embedded_imgs
            except Exception as e:
                print(f"⚠️ Image extraction failed on Page {page_number} in {file_name}: {e}")

            structured_output["pages"].append(page_data)

        with open(save_path, "w") as f:
            json.dump(structured_output, f, indent=4)

        all_pdfs_data[pdf_id] = structured_output

    return all_pdfs_data

# Utility Functions
def clean_text(text: str) -> str:
    """Cleans up extracted text by normalizing whitespace and removing excessive newlines."""
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def df_to_dict_records(df: pd.DataFrame):
    """Converts a DataFrame into list-of-dictionaries for JSON serialization."""
    return df.to_dict(orient="records")
