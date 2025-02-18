import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import re
from google import genai

# Initialize the Gemini client
client = genai.Client(api_key="AIzaSyC5FUT2l7ApdCB19sE2i_ZxWb11BJkwubY")


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
            response = client.models.generate_content(model=model, contents=prompt)
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

def load_step3_data_to_df(step3_data_path="processed_data/step3_data.json"):
    """Loads Step 3 JSON data and creates a DataFrame with chunk_id, page_range, chunk_text, qa_pairs, keywords."""
    try:
        with open(step3_data_path, "r") as f:
            step3_data = json.load(f)
        print("📂 Step 3 Data Loaded Successfully!")
    except Exception as e:
        print(f"❌ Failed to load {step3_data_path}: {e}")
        return pd.DataFrame()

    records = []
    for entry in step3_data:
        records.append({
            "chunk_id": entry.get("chunk_id", "Unknown"),
            "page_range": entry.get("page_range", "Unknown"),
            "chunk_text": entry.get("chunk_text", "").strip(),
            "qa_pairs": entry.get("qa_pairs", []),
            "keywords": entry.get("keywords", [])
        })
    
    df = pd.DataFrame(records)
    print(f"🔍 Created DataFrame with {len(df)} chunks.")
    return df


def find_similar_questions(user_question, df, threshold=0.7):
    """Finds the most similar question from the stored Q&A pairs using TF-IDF similarity."""
    all_qa_pairs = []
    for _, row in df.iterrows():
        for qa in row["qa_pairs"]:
            all_qa_pairs.append(qa)
    
    if not all_qa_pairs:
        return None

    vectorizer = TfidfVectorizer()
    questions = [qa["question"] for qa in all_qa_pairs]
    tfidf_matrix = vectorizer.fit_transform(questions + [user_question])
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    best_match_idx = similarities.argmax()
    if similarities[best_match_idx] >= threshold:
        return all_qa_pairs[best_match_idx]["answer"]
    return None


def find_relevant_chunks(user_question, df, top_n=5):
    """Finds the most relevant document chunks using TF-IDF similarity and keyword matching."""
    if df.empty:
        return []

    vectorizer = TfidfVectorizer()
    chunk_texts = df["chunk_text"].tolist()
    tfidf_matrix = vectorizer.fit_transform(chunk_texts + [user_question])
    
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    df["similarity_score"] = similarities
    df["has_numbers"] = df["chunk_text"].apply(lambda text: any(char.isdigit() for char in text))
    df["has_keywords"] = df.apply(lambda row: any(kw.lower() in row["chunk_text"].lower() for kw in row["keywords"]), axis=1)
    
    sorted_df = df.sort_values(by=["has_keywords", "has_numbers", "similarity_score"], ascending=False)
    
    return sorted_df.head(top_n)[["chunk_id", "page_range", "chunk_text"]].to_dict(orient="records")


def answer_user_question(user_question, df):
    """Answers user queries using stored Q&A, relevant document chunks, and Gemini API."""
    print(f"\n❓ **User Question:** {user_question}\n")
    if isinstance(df, list):
        df = pd.DataFrame(df)
    similar_answer = find_similar_questions(user_question, df)
    if similar_answer:
        print("✅ Found a similar question in stored Q&A pairs!\n")
        return similar_answer, []
    
    relevant_chunks = find_relevant_chunks(user_question, df, top_n=5)
    
    if relevant_chunks:
        print(f"📄 Found {len(relevant_chunks)} relevant document chunks. Sending to Gemini...\n")
        
        context = "\n\n".join([f"(Chunk {chunk['chunk_id']}, Page {chunk['page_range']}) {chunk['chunk_text']}" for chunk in relevant_chunks])
        
        prompt = f"""
        You are an AI assistant. Answer the following question using only the provided document context.
        
        **User Question:** {user_question}
        **Document Context:** 
        {context}
        
        **Strictly return JSON output in the following format:**
        ```json
        {{
            "answer": "Your generated response here.",
            "source": "List of extracted sources from the text"
        }}
        ```
        """
        
        response_json = call_gemini_with_backoff(prompt)
        return response_json.get("answer", "Error: No valid response from Gemini."), response_json.get("source", [])
