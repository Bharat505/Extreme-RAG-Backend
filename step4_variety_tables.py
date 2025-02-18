import json
import time
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import google.generativeai as genai  # Corrected import

# Initialize the Gemini client
client=genai.configure(api_key="AIzaSyC5FUT2l7ApdCB19sE2i_ZxWb11BJkwubY")  # Corrected client setup

# Define output directories
PROCESSED_DIR = "processed_data"
IMAGE_DIR = os.path.join(PROCESSED_DIR, "table_visualizations")
os.makedirs(IMAGE_DIR, exist_ok=True)

def clean_json_response(response_text: str) -> str:
    """Removes markdown and ensures valid JSON output."""
    response_text = response_text.strip()
    response_text = re.sub(r"^```json", "", response_text)  # Remove markdown json block
    response_text = re.sub(r"```$", "", response_text)  # Remove closing markdown
    return response_text

def clean_numeric_values(df):
    """ Cleans numeric values in a DataFrame by removing symbols and converting to numeric."""
    for col in df.columns[1:]:  # Skip first column (categorical)
        # ✅ Ensure column is a string before applying `.str.replace()`
        if df[col].dtypes == 'O':  # 'O' stands for object (string-like columns)
            df[col] = df[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True)  # Remove non-numeric chars
        
        # ✅ Convert column to numeric (coerce invalid values to NaN)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def call_gemini_with_backoff(prompt, max_retries=5, initial_delay=5):
    """
    Calls Gemini API with exponential backoff to handle rate limits (429 errors) and ensures JSON response.
    """
    for attempt in range(max_retries):
        try:
            response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)  # ✅ Fixed API Call
            raw_text = response.text  # Extract response text directly

            print(f"🔍 Gemini Raw Response (Attempt {attempt + 1}): {raw_text}")  # ✅ Debugging output

            # Convert JSON response
            return json.loads(raw_text)

        except json.JSONDecodeError:
            print(f"❌ JSON Decode Error (Attempt {attempt + 1}) - Response: {raw_text}")
            time.sleep(initial_delay * (2 ** attempt))  # Exponential backoff
        
        except RuntimeError as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait_time = initial_delay * (2 ** attempt)
                print(f"⚠️ API Quota Exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e  # Raise any other runtime errors immediately

    print("❌ Max retries reached. Could not process request.")
    return {}

def process_forced_variety_tables(df_step3, output_tables="processed_data/step4_forced_variety_tables.json"):
    """
    Processes tables from Step 3 output and ensures visualization variety.
    """
    extracted_tables = []
    
    for entry in df_step3:
        pdf_id = entry["pdf_id"]
        chunk_id = entry["chunk_id"]
        tables = entry.get("tables", [])

        if not tables:
            print(f"⚠️ No tables found in chunk {chunk_id} of PDF {pdf_id}. Skipping...")
            continue

        for idx, table in enumerate(tables, start=1):
            if not isinstance(table, dict) or "columns" not in table or "rows" not in table:
                print(f"⚠️ Table {idx} in chunk {chunk_id} is improperly formatted. Skipping...")
                continue

            table["table_id"] = f"{pdf_id}_{chunk_id}_{idx}"
            table["pdf_id"] = pdf_id
            table["chunk_id"] = chunk_id
            extracted_tables.append(table)

    if not extracted_tables:
        print("❌ No valid tables extracted from Step 3 data.")
        return

    print(f"✅ Found {len(extracted_tables)} tables. Sending to Gemini for variety filtering...")

    batch_size = 10  # ✅ Process in batches to improve response quality
    selected_tables = []

    for i in range(0, len(extracted_tables), batch_size):
        batch_tables = extracted_tables[i:i + batch_size]

        prompt = f"""
        You are an expert financial analyst and visualization expert. Your task is to:
        
        1️⃣ **Rank and select the top 8-10 most insightful tables** from the provided dataset.
        2️⃣ **Assign each table a meaningful and business-relevant title.**
        3️⃣ **Strictly ensure the following visualization variety rules:**
           - **Maximum 2 Line Charts per batch.**
           - **At least 3 other chart types must be included: Bar, Pie, Heatmap, Scatter, Stacked Area, etc.**
        
        4️⃣ **Ensure the response follows the exact JSON format below.**
        
        **Required JSON format:**
        {{
            "selected_tables": [
                {{
                    "table_id": "{batch_tables[0]['table_id']}",
                    "pdf_id": "{batch_tables[0]['pdf_id']}",
                    "chunk_id": "{batch_tables[0]['chunk_id']}",
                    "title": "Revenue Breakdown by Region (2020-2023)",
                    "recommended_visualization": "bar chart",
                    "insight": "Highlights revenue distribution across different regions.",
                    "table_data": {{
                        "columns": {json.dumps(batch_tables[0].get('columns', []), indent=2)},
                        "rows": {json.dumps(batch_tables[0].get('rows', []), indent=2)}
                    }}
                }}
            ]
        }}
        
        **Extracted & Generated Tables:**
        {json.dumps(batch_tables, indent=2)}
        """
        
        print(f"\n🚀 Sending Batch {i // batch_size + 1} to Gemini...")
        results = call_gemini_with_backoff(prompt)

        print("\n🔍 Gemini Raw Response:", results)  # Debugging Output

        if results and "selected_tables" in results:
            selected_tables.extend(results["selected_tables"])
        else:
            print(f"⚠️ Empty or invalid response received for batch {i // batch_size + 1}.")

    # ✅ Select only the top 8-10 tables ranked by importance
    selected_tables = sorted(selected_tables, key=lambda x: x.get("insight", ""), reverse=True)[:10]

    if selected_tables:
        print(f"✅ Successfully processed {len(selected_tables)} tables with varied visualizations.")
        with open(output_tables, "w") as f:
            json.dump(selected_tables, f, indent=4)
        print(f"✅ Selected Tables with Titles & Insights saved to {output_tables}")
    else:
        print("❌ No valid tables generated.")


def visualize_top_tables(output_tables="processed_data/step4_forced_variety_tables.json"):
    """
    Reads selected tables JSON and creates interactive visualizations based on their recommended type.
    """
    if not os.path.exists(output_tables):
        print(f"❌ Visualization failed. {output_tables} not found.")
        return

    try:
        with open(output_tables, "r") as f:
            results = json.load(f)
        print(f"📂 Loaded {len(results)} tables from {output_tables}")
    except Exception as e:
        print(f"❌ Failed to load {output_tables}: {e}")
        return
    
    if not results:
        print("❌ No tables found for visualization.")
        return

    for table in results[:5]:
        title = table['title']
        viz_type = table['recommended_visualization']
        table_data = table['table_data']
        table_id = table['table_id']

        if not isinstance(table_data, dict) or "columns" not in table_data or "rows" not in table_data:
            print(f"⚠️ Skipping table {table_id} - Incorrect data format.")
            continue

        df = pd.DataFrame(table_data["rows"], columns=table_data["columns"])
        df = clean_numeric_values(df)

        image_path = os.path.join(IMAGE_DIR, f"{table_id}.html")
        fig = px.bar(df, x=df.columns[0], y=df.columns[1:], title=title)
        pio.write_html(fig, image_path)
        print(f"✅ Visualization saved: {image_path}")
