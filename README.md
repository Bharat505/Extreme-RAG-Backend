# 🚀 Smart PDF Insights Assistant - AI-Powered Document Analysis

**Smart PDF Insights Assistant** is a state-of-the-art AI-driven system that extracts, summarizes, compares, and answers questions from PDFs using **Retrieval-Augmented Generation (RAG)**, powered by leading LLMs like **Gemini** and **GPT**.

## 🌟 Key Features
- 🔍 **Instantly extracts key insights** from PDFs
- 📄 **Generates structured summaries & document comparisons**
- ❓ **Auto-generates the top questions from documents**
- 📊 **Transforms tables into interactive visualizations**
- 🤖 **Provides AI-powered answers with source citations**
- ⚡ **Processes multiple PDFs simultaneously for efficiency**

## 🚀 Live Application
- **Frontend (Deployed on Vercel):** [Smart PDF Insights Assistant](#) *(Insert live link when available)*
- **Backend:** Running locally *(Cloud deployment planned soon)*

## 📌 GitHub Repositories
- **Frontend Repo:** [Extreme-RAG-Frontend](https://github.com/Bharat505/Extreme-RAG-Frontend)
- **Backend Repo:** [Extreme-RAG-Backend](https://github.com/Bharat505/Extreme-RAG-Backend)

## 📖 Table of Contents
1. [Setup & Installation](#-setup--installation)
2. [Running the Application](#-running-the-application)
3. [How It Works (LLM-Powered Processing)](#-how-it-works-llm-powered-processing)
4. [System Architecture & Workflow](#-system-architecture--workflow)
5. [Future Enhancements](#-future-enhancements)
6. [Contributors & Support](#-contributors--support)

---

## 🛠 Setup & Installation
This guide provides a step-by-step walkthrough for setting up both the **backend (FastAPI)** and **frontend (React)** for local development.

### 1️⃣ Backend Setup (FastAPI)
#### Step 1: Clone the Repository
```sh
git clone https://github.com/Bharat505/Extreme-RAG-Backend.git  
cd Extreme-RAG-Backend  
```
#### Step 2: Create & Activate a Virtual Environment
```sh
python -m venv env  
source env/bin/activate  # Mac/Linux  
env\Scripts\activate  # Windows  
```
#### Step 3: Install Dependencies
```sh
pip install -r requirements.txt  
```
#### Step 4: Start the Backend Server
```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload  
```
📍 The backend will now be accessible at: [http://127.0.0.1:8000/docs/](http://127.0.0.1:8000/docs/) *(Swagger API UI)*

---

### 2️⃣ Frontend Setup (React)
#### Step 1: Clone the Frontend Repository
```sh
git clone https://github.com/Bharat505/Extreme-RAG-Frontend.git  
cd Extreme-RAG-Frontend  
```
#### Step 2: Install Dependencies
```sh
npm install  
```
#### Step 3: Start the Frontend
```sh
npm start  
```
📍 The frontend will be available at: [http://localhost:3000/](http://localhost:3000/)

---

## 🚀 Running the Application
### Step-by-Step Workflow
1. **Upload PDFs** via the frontend
2. **Automated AI processing begins** instantly
3. **Track progress** with real-time status updates
4. **Explore structured insights** across multiple sections:
   - 📌 **Final Summaries**
   - 🔎 **Comparisons & Key Differences**
   - ❓ **Top Questions Generated**
   - 📊 **Interactive Table Visualizations**
5. **Ask AI-powered queries** and receive **source-cited responses**

---

## 🔍 How It Works (LLM-Powered Processing)
Our system follows a **five-step AI-powered process** using optimized LLM prompts to generate structured document insights.

### 1️⃣ Step 1: Extraction (Text, Tables, Images)
- Utilizes **PyMuPDF, Camelot, and Tesseract OCR**
- Extracts **text, tables, and embedded images**

### 2️⃣ Step 2: Chunking (Splitting Text for AI Processing)
- Segments PDFs into **logical sections** (headings, paragraphs, structured tables)
- Ensures **accurate page referencing** for efficient retrieval

### 3️⃣ Step 3: Summarization & Q&A Generation
- AI-powered **chunk-level summarization**
- Extracts **question-answer pairs** for each chunk
- Detects **tables & converts them into structured JSON**

### 4️⃣ Step 4: Document-Level Summarization & Comparisons
- Aggregates **chunk-level summaries** into **comprehensive document overviews**
- Compares multiple PDFs (e.g., **Annual Reports vs. White Papers**)
- Identifies **key insights, trends, and differences**

### 5️⃣ Step 5: AI-Powered User Query System
- Users **ask free-form questions**
- The system **retrieves relevant document chunks**
- AI generates **precise, source-cited answers** (including file name & page number)

---

## 🏗 System Architecture & Workflow
### 🌐 High-Level Architecture
#### 📌 **Frontend (React)**
- **Uploads & manages PDFs**
- **Monitors processing steps**
- **Displays AI-generated results**
- **Handles interactive Q&A retrieval**

#### 📌 **Backend (FastAPI)**
- **Extracts structured content** (text, tables, images)
- **Generates document summaries & insights**
- **Performs AI-powered comparisons**
- **Manages retrieval-augmented answers**

### ⚙ **Backend Processing Workflow**
1. User **uploads PDFs** → FastAPI **extracts text & tables**
2. Extracted data is **chunked & processed** by **Gemini/GPT**
3. AI-powered **summaries, comparisons & Q&A** are generated
4. User **queries are answered** using intelligent document retrieval

---

## 🚀 Future Enhancements
- ☁ **Deploy Backend on Cloud** (AWS/GCP/Azure)
- 📂 **Support Additional Document Formats** (Word, PPT, Excel)
- 🔗 **Integrate with External Knowledge Bases**
- 🎯 **Fine-Tune AI for Industry-Specific Analysis**
- 📊 **Enhance Data Visualizations for Deeper Insights**

---

## 👥 Contributors & Support
### Developed by:
👤 **Bharat505** (Lead Developer)

We encourage **open-source contributions**! 🚀
- Submit **pull requests** or **open issues** to improve the tool.

### 💡 Need Help?
📩 **GitHub Issues** or 📧 **Email Support**

---

## 🎯 Final Thoughts
The **Smart PDF Insights Assistant** transforms **static PDFs into dynamic, AI-driven insights**.

It integrates **automated summarization, document comparisons, top questions, interactive tables, and AI-powered responses** into a unified, cutting-edge tool.

✅ **Technology Stack:**
- **FastAPI** (Backend)
- **React** (Frontend)
- **LLMs (Gemini/GPT)**
- **RAG-Based Retrieval**
- **AI-Enhanced Summarization & Q&A**

🚀 **Enjoy using Smart PDF Insights Assistant!**

