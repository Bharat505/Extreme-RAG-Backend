import os
import re
import json
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor  # For parallel processing
import google.generativeai as genai  # Corrected import

# Initialize the Gemini client
genai.configure(api_key="AIzaSyC5FUT2l7ApdCB19sE2i_ZxWb11BJkwubY")  # Corrected client setup

# ----------------- Utility Functions -----------------

def clean_json_response(response_text: str) -> str:
    """Removes markdown (e.g., ```json ... ```) and ensures valid JSON output."""
    response_text = response_text.strip()
    response_text = re.sub(r"^```json\s*", "", response_text)  # Remove starting ```json
    response_text = re.sub(r"```$", "", response_text)  # Remove ending ```
    return response_text

def call_gemini(prompt: str, model: str = "gemini-2.0-flash", max_retries: int = 3) -> dict:
    """Calls the Gemini API and ensures valid JSON output."""
    for attempt in range(max_retries):
        try:
            response = genai.GenerativeModel(model).generate_content(prompt)  # ✅ Fixed API call
            raw_text = clean_json_response(response.text)
            return json.loads(raw_text)
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON Response (Attempt {attempt + 1}): {response.text[:200]}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise ValueError("API returned invalid JSON")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise RuntimeError(f"Error calling Gemini API: {e}")

def call_gemini_with_backoff(prompt, max_retries=3):
    """ Calls Gemini API with backoff and ensures JSON response. """
    for attempt in range(max_retries):
        try:
            response = call_gemini(prompt)
            if isinstance(response, dict):
                return response
            return json.loads(response)
        except (json.JSONDecodeError, TypeError):
            print(f"❌ Invalid JSON Response (Attempt {attempt + 1})")
            if attempt < max_retries - 1:
                continue
            return {"answer": "Error: Unable to process request."}
    return {"answer": "Error: Unable to process request."}


# ----------------- LLM-Based Summarization, Q&A, and Table Extraction -----------------

def summarize_text(text: str) -> dict:
    """Generates a structured summary and extracts relevant keywords."""
    summary_prompt = f"""You are an AI assistant. Analyze the text below and return:
    
1. A **structured summary** with:
   - **Introduction**: A concise overview (1–2 sentences).
   - **Main Points**: Key details (brief paragraph).
   - **Conclusion**: Final takeaway (1–2 sentences).

2. A list of **important keywords** representing the core themes.

**⚠️ Output strict JSON only. No extra text or formatting.**  

TEXT: {text}
"""
    return call_gemini(summary_prompt)

def extract_qa_and_tables(text: str) -> dict:
    """Generates relevant Q&A pairs and extracts tables from the text chunk."""
    qa_prompt = f"""You are an AI assistant analyzing the provided text. 

Please return two structured outputs in JSON format:

1️⃣ **Question-Answer Pairs (Q&A):**
   - Generate **insightful questions and concise answers** from the text.
   - Answers should be **strictly based on the text**.

2️⃣ **Tables (if any exist in the text)**:
   - Detect structured **tabular data**.
   - Extract and return tables in JSON format, structured as:
     ```json
     {{
       "table_id": int,
       "columns": ["col1", "col2", ...],
       "rows": [
         ["value1", "value2", ...],
         ["value3", "value4", ...]
       ]
     }}
     ```
   - If **no table exists**, return an empty list.

⚠️ **Output strict JSON only. No extra text or formatting.**  

TEXT:  
{text}
"""
    return call_gemini(qa_prompt)

# ----------------- Processing Each Chunk (Parallel Processing Enabled) -----------------

def process_chunk(chunk):
    """Processes a single chunk: Summarization, Q&A, and Table Extraction."""
    pdf_id = chunk["pdf_id"]
    file_name = chunk["file_name"]
    chunk_id = chunk["chunk_id"]
    page_range = chunk["page_range"]
    chunk_text = chunk["chunk_text"]

    try:
        summary_data = summarize_text(chunk_text)
    except Exception as e:
        print(f"❌ Error processing summarization for chunk {chunk_id}: {e}")
        summary_data = {"summary": {}, "keywords": []}

    try:
        qa_table_data = extract_qa_and_tables(chunk_text)
        qa_data = qa_table_data.get("qa_pairs", [])  # Extract Q&A
        extracted_tables = qa_table_data.get("tables", [])  # Extract Tables
    except Exception as e:
        print(f"❌ Error processing Q&A and table extraction for chunk {chunk_id}: {e}")
        qa_data, extracted_tables = [], []

    return {
        "pdf_id": pdf_id,
        "file_name": file_name,
        "chunk_id": chunk_id,
        "page_range": page_range,
        "chunk_text": chunk_text,
        "summarization": summary_data.get("summary", {}),
        "keywords": summary_data.get("keywords", []),
        "qa_pairs": qa_data,
        "tables": extracted_tables  # Store extracted tables separately
    }

# ----------------- Running the Pipeline (Parallel Processing Enabled) -----------------

def process_step3(chunked_results):
    """Runs the processing pipeline for Step 3."""
    results = []
    with ThreadPoolExecutor() as executor:  # Enables multi-threaded processing for speed
        results = list(executor.map(process_chunk, chunked_results))
    df = pd.DataFrame(results)
    save_step3_data(df)
    return df

def save_step3_data(df, output_dir="processed_data"):
    """Saves processed results (summaries, Q&A, tables) from Step 3 using DataFrame."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df["summarization"] = df["summarization"].apply(lambda x: x if isinstance(x, dict) else {"introduction": "", "main_points": "", "conclusion": ""})
    df["keywords"] = df["keywords"].apply(lambda x: x if isinstance(x, list) else [])
    df["qa_pairs"] = df["qa_pairs"].apply(lambda x: x if isinstance(x, list) and x else [{"question": "No question generated", "answer": "N/A"}])
    df["tables"] = df["tables"].apply(lambda x: x if isinstance(x, list) else [])
    
    df.to_json(os.path.join(output_dir, "step3_data.json"), orient="records", indent=4)
    print("✅ Step 3 data saved successfully!")
