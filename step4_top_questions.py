import json
import time
import re
import google.generativeai as genai  # Corrected import

# Initialize the Gemini client
genai.configure(api_key="AIzaSyC5FUT2l7ApdCB19sE2i_ZxWb11BJkwubY")  # Corrected client setup

def clean_json_response(response_text: str) -> str:
    """Cleans up raw JSON response from Gemini, ensuring proper formatting."""
    response_text = response_text.strip()
    response_text = re.sub(r'^```json\s*', '', response_text)  # Remove starting ```json
    response_text = re.sub(r'```$', '', response_text)  # Remove ending ```
    return response_text

def call_gemini_with_backoff(prompt, max_retries=5, initial_delay=5):
    """
    Calls Gemini API with exponential backoff to handle rate limits (429 errors) and ensures JSON response.
    """
    for attempt in range(max_retries):
        try:
            response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)  # ✅ Fixed API Call
            raw_text = response.text  # Extract response text directly

            print(f"🔍 Gemini Raw Response (Attempt {attempt + 1}):", raw_text)  # ✅ Debugging output

            # Convert JSON response
            return json.loads(raw_text)

        except json.JSONDecodeError:
            print(f"❌ JSON Decode Error (Attempt {attempt + 1}) - Response:\n{raw_text}")
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

def process_top_questions(df_step3, output_qa="processed_data/step4_top_questions.json"):
    """
    Extracts top questions directly from step3 data and calls Gemini API for ranking.
    """
    pdf_questions = {}
    all_questions = set()

    for entry in df_step3:
        pdf_id = entry["pdf_id"]
        pdf_name = entry.get("file_name", pdf_id)
        qa_pairs = entry.get("qa_pairs", [])
        
        questions = [qa["question"] for qa in qa_pairs if "question" in qa]
        pdf_questions[pdf_id] = {
            "pdf_name": pdf_name,
            "questions": questions
        }
        all_questions.update(questions)

    if not all_questions:
        print("⚠️ No questions found in Step 3 data!")
        return

    print(f"🔍 Found {len(all_questions)} unique questions. Sending to Gemini...")

    prompt = f"""
    You are an AI knowledge assistant. Your task is to:
    1️⃣ Select the **top 10 most relevant questions overall** from the provided dataset.
    2️⃣ Identify the **top 10 most insightful questions per document**.
    3️⃣ Prioritize questions that:
       - Enable deep analytical insights.
       - Are highly relevant to document summaries.
       - Are non-repetitive and meaningful.
    4️⃣ **STRICTLY return only valid JSON in the format below.**
    
    🚫 **Do NOT include explanations, markdown, or extra text.**
    🚫 **DO NOT return JSON inside a markdown block.**
    
    **Required JSON format:**
    {{
        "top_10_overall_questions": [
            "Example overall question 1",
            "Example overall question 2"
        ],
        "top_10_per_pdf_questions": {{
            "pdf_id_1": {{
                "pdf_name": "Example PDF 1",
                "top_questions": ["Example question A", "Example question B"]
            }},
            "pdf_id_2": {{
                "pdf_name": "Example PDF 2",
                "top_questions": ["Example question X", "Example question Y"]
            }}
        }}
    }}
    
    **Extracted Questions:**
    {json.dumps(pdf_questions, indent=2)}
    """
    
    results = call_gemini_with_backoff(prompt)
    
    print("\n🔍 Gemini Raw Response:", results)
    
    if results and "top_10_overall_questions" in results and "top_10_per_pdf_questions" in results:
        with open(output_qa, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"✅ Top 10 Overall & Per-PDF Questions saved to {output_qa}")
    else:
        print("❌ No valid questions generated.")
