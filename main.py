from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import json
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import glob

from step1_extract import extract_full_content_structured
from step2_chunking import process_chunking
from step3_summarization import process_step3, save_step3_data
from step4_comparison import generate_final_summaries_and_comparisons
from step4_top_questions import process_top_questions
from step4_variety_tables import process_forced_variety_tables, visualize_top_tables
from query_processing import load_step3_data_to_df, answer_user_question

# Initialize FastAPI
app = FastAPI()

# ✅ Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_pdfs")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
TABLE_VISUALS_DIR = os.path.join(PROCESSED_DATA_DIR, "table_visualizations")

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(TABLE_VISUALS_DIR, exist_ok=True)

# ✅ Serve Static Files (Required for serving Plotly HTML visualizations)
app.mount("/static", StaticFiles(directory=PROCESSED_DATA_DIR), name="static")

# ✅ Keep `df_step3` Live for Queries
df_step3 = None

# ✅ Function to retry failed steps (Max 2 retries)
def execute_with_retries(task_func, *args, max_retries=2):
    for attempt in range(max_retries):
        try:
            print(f"🚀 Attempt {attempt + 1}/{max_retries} - Running {task_func.__name__}...")
            return task_func(*args)  
        except Exception as e:
            print(f"❌ Error in {task_func.__name__}: {e}")
            if attempt < max_retries - 1:
                print(f"🔄 Retrying {task_func.__name__}...")
                time.sleep(2)
            else:
                print(f"❌ Max retries reached for {task_func.__name__}. Skipping.")
                return None

@app.post("/run_pipeline/")
async def run_pipeline(files: list[UploadFile] = File(...)):
    """Processes multiple uploaded PDF files through all pipeline steps."""
    global df_step3
    file_paths = []

    for file in files:
        file_path = os.path.join(PROCESSED_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
        file_paths.append(file_path)

    # ✅ Step 1 & 2: Extract and Chunk Data
    extracted_data = execute_with_retries(extract_full_content_structured, file_paths)
    chunked_data = execute_with_retries(process_chunking, extracted_data)

    if not isinstance(chunked_data, list):  
        raise ValueError("❌ Error: chunked_data must be a list of dictionaries")

    # ✅ Step 3: Summarization & Processing
    df_step3 = execute_with_retries(process_step3, chunked_data)
    save_step3_data(df_step3, output_dir=PROCESSED_DATA_DIR)
    print("✅ Step 3 Data Saved!")

    if isinstance(df_step3, pd.DataFrame):
        df_step3 = df_step3.to_dict(orient="records")

    # ✅ Step 4: Run in Parallel (Summaries, Questions, Tables)
    print("🚀 Running Step 4 in Parallel...")
    tasks = {
        "final_summaries": (generate_final_summaries_and_comparisons, df_step3, 
                            os.path.join(PROCESSED_DATA_DIR, "final_summaries.json"), 
                            os.path.join(PROCESSED_DATA_DIR, "comparisons.json")),
        "top_questions": (process_top_questions, df_step3, 
                          os.path.join(PROCESSED_DATA_DIR, "step4_top_questions.json")),
        #"forced_variety_tables": (process_forced_variety_tables, df_step3, 
         #                         os.path.join(PROCESSED_DATA_DIR, "step4_forced_variety_tables.json")),
        "visualizations": (visualize_top_tables,)
    }

    with ThreadPoolExecutor() as executor:
        future_tasks = {executor.submit(execute_with_retries, *params): name for name, params in tasks.items()}
        
        for future in as_completed(future_tasks):
            task_name = future_tasks[future]
            try:
                result = future.result()
                print(f"✅ {task_name} completed successfully!")
            except Exception as e:
                print(f"❌ {task_name} failed: {e}")

    print("✅ Step 4 Completed!")

    return {"message": "Pipeline completed successfully!"}

@app.get("/get-final-summaries/")
def get_final_summaries():
    """Returns final summaries JSON content as soon as it's available."""
    path = os.path.join(PROCESSED_DATA_DIR, "final_summaries.json")
    return read_json_file(path)

@app.get("/get-comparisons/")
def get_comparisons():
    """Returns comparisons JSON content as soon as it's available."""
    path = os.path.join(PROCESSED_DATA_DIR, "comparisons.json")
    return read_json_file(path)

@app.get("/get-top-questions/")
def get_top_questions():
    """Returns top questions JSON content."""
    path = os.path.join(PROCESSED_DATA_DIR, "step4_top_questions.json")
    return read_json_file(path)

@app.get("/get-table-visuals/")
def get_table_visuals():
    """Returns available table visualizations as URLs."""
    if not os.path.exists(TABLE_VISUALS_DIR):
        return {"error": "No table visualizations found."}

    visualization_files = [
        f"http://127.0.0.1:8000/get-table-visual/{file_id}"
        for file_id in os.listdir(TABLE_VISUALS_DIR) if file_id.endswith(".html")
    ]

    return {"table_visuals": visualization_files if visualization_files else "No table visualizations available."}

@app.get("/get-table-visual/{file_id}")
async def get_table_visual(file_id: str):
    """Returns the requested table visualization .html file."""
    visual_path = os.path.join(TABLE_VISUALS_DIR, f"{file_id}.html")

    if not os.path.exists(visual_path):
        possible_files = glob.glob(os.path.join(TABLE_VISUALS_DIR, f"{file_id}*"))
        if possible_files:
            visual_path = possible_files[0]  

    if os.path.exists(visual_path):
        return FileResponse(visual_path, media_type="text/html")
    
    return {"error": "Visualization file not found"}

@app.post("/ask-question/")
def ask_question_endpoint(payload: dict):
    """Handles user queries by retrieving answers from the live document data."""
    global df_step3

    if df_step3 is None:
        raise HTTPException(status_code=500, detail="Step 3 data is not available yet.")

    question = payload.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    answer, sources = answer_user_question(question, df_step3)
    return {"answer": answer, "source": sources}

# ✅ Utility function to read JSON files safely
def read_json_file(filepath):
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}
    
    with open(filepath, "r") as file:
        return json.load(file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
