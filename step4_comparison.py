import json
import time
import re
from google import genai

# Initialize the Gemini client
client = genai.Client(api_key="AIzaSyC5FUT2l7ApdCB19sE2i_ZxWb11BJkwubY")

# ----------------- Utility Functions -----------------

def clean_json_response(response_text: str) -> str:
    """Removes markdown (e.g., ```json ... ```) and ensures valid JSON output."""
    response_text = response_text.strip()
    response_text = re.sub(r"^```json\s*", "", response_text)  # Remove starting ```json
    response_text = re.sub(r"```$", "", response_text)  # Remove ending ```
    return response_text

def call_gemini_with_backoff(prompt, max_retries=5, initial_delay=5):
    """
    Calls Gemini API with exponential backoff to handle rate limits (429 errors) and ensures JSON response.
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            raw_text = clean_json_response(response.text)
            
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                print(f"⚠️ Invalid JSON format received. Retrying... (Attempt {attempt + 1})")
                time.sleep(initial_delay * (2 ** attempt))
        
        except json.JSONDecodeError:
            print(f"❌ JSON Decode Error (Attempt {attempt + 1})")
            time.sleep(initial_delay * (2 ** attempt))
        except RuntimeError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                wait_time = initial_delay * (2 ** attempt)
                print(f"⚠️ API Quota Exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    
    print("❌ Max retries reached. Could not process request.")
    return {}

def generate_final_summaries_and_comparisons(df_step3, output_summary="processed_data/final_summaries.json", output_comparison="processed_data/comparisons.json"):
    """
    Loads step 3 summaries from a DataFrame, calls Gemini for refined summaries & comparisons, and saves the results.
    """
    pdf_summaries = {}
    
    for entry in df_step3:
        pdf_id = entry["pdf_id"]
        chunk_summaries = [json.dumps(s.get("summarization", {"introduction": "", "main_points": "", "conclusion": ""})) for s in df_step3 if s["pdf_id"] == pdf_id]
        pdf_summaries[pdf_id] = "\n".join(chunk_summaries)
    
    combined_summary_texts = "\n\n".join(
        [f"PDF: {pdf_id}\nSummary: {summary}" for pdf_id, summary in pdf_summaries.items()]
    )
    
    prompt = f"""
    You are an AI assistant analyzing multiple documents. Strictly return JSON output with the following structure:
    ```json
    {{
        "final_summaries": {{
            "pdf_id_1": {{ "Title & Subject": "", "Introduction": "", "Key Findings": "", "Trends & Patterns": "", "Opportunities & Risks": "", "Data-Backed Evidence": "", "Conclusion & Recommendations": "" }},
            "pdf_id_2": {{ "Title & Subject": "", "Introduction": "", "Key Findings": "", "Trends & Patterns": "", "Opportunities & Risks": "", "Data-Backed Evidence": "", "Conclusion & Recommendations": "" }}
        }},
        "comparison": {{
            "Common Insights": "",
            "Key Differences": "",
            "Contradictions or Conflicting Data": "",
            "Relevance to Business Strategy": ""
        }}
    }}
    ```
    Do not include any additional text or explanations outside of this JSON format.
    
    TEXT:
    {combined_summary_texts}
    """
    
    results = call_gemini_with_backoff(prompt)
    
    if not isinstance(results, dict):
        print("⚠️ Warning: Unexpected response type from API. Using fallback structure.")
        results = {}
    
    final_summaries = results.get("final_summaries", {})
    comparison = results.get("comparison", {})
    
    with open(output_summary, "w") as f:
        json.dump(final_summaries, f, indent=4)
    
    with open(output_comparison, "w") as f:
        json.dump(comparison, f, indent=4)
    
    print(f"✅ Final summaries saved to {output_summary}")
    print(f"✅ Comparison insights saved to {output_comparison}")
