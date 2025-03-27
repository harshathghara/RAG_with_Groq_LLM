# 🔍 RAG-Based PDF Search Engine with LLM (Groq)

# Project Overview

RAG-Based PDF Search Engine with LLM (Groq)

This project is designed to search for relevant information within PDFs and generate intelligent responses using Groq's LLM. It follows a Retrieval-Augmented Generation (RAG) approach, which enhances the capabilities of LLMs by retrieving precise information from indexed document embeddings.

Here's how it works:

Text Extraction 📝 – Extracts text from PDFs using PyMuPDF.

Embedding Creation 🔢 – Converts the text into embeddings using SentenceTransformers.

Vector Search 🔍 – Stores & retrieves embeddings via FAISS for fast lookups.

LLM Integration 🤖 – Uses Groq LLM to generate meaningful answers based on search results.

This makes it an efficient document retrieval system, helping users ask questions and get contextual responses from stored PDFs. Ideal for research, legal documents, or any large text-based datasets.

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** that allows users to **search for information in PDF documents** using a **vector search engine (FAISS)** and an **LLM-powered response generation (Groq)**.

## 🚀 Features
- ✅ **Extracts text from PDFs** using `PyMuPDF`
- ✅ **Generates embeddings** using `SentenceTransformers`
- ✅ **Stores and retrieves vectors** using `FAISS`
- ✅ **Processes natural language queries**
- ✅ **Uses Groq LLM to generate responses**

---

## 🏗️ Project Structure

📂 RAG_PDF_Search  
│── 📂 data/  
│   └── resume.pdf  # Your input PDF file 
│   └── Test.pdf  # Your input PDF file  
│  
│── 📂 processed_text/  
│   └── resume.txt  # Extracted text from PDF  
|   └── Test.txt  # Extracted text from PDF 
│  
│── 📂 embeddings/  
│   ├── faiss_index  # FAISS vector database  
│   ├── faiss_index_metadata.npy  # Stores original text chunks 
│   ├── faiss_index_for_Test  # FAISS vector database  
│   ├── faiss_index_metadata_for_Test.npy  # Stores original text chunks  
│  
│── 📂 src/  
│   ├── extract_text.py  # Extract text from PDF  
│   ├── generate_embeddings.py  # Generate & store embeddings in FAISS  
│   ├── query_engine.py  # Interactive search engine  
│   ├── llm_response.py   # Integrates LLM for responses
│  
│── 📜 requirements.txt  # Dependencies  
│── 📜 README.md  # Project documentation  


---

## 🔧 Setup Instructions
### 
1️⃣ Install Dependencies

pip install -r requirements.txt

2️⃣ Extract Text from PDFs

python src/extract_text.py

3️⃣ Generate FAISS Embeddings

python src/generate_embeddings.py

4️⃣ Run the Query Engine

python src/query_engine.py

5️⃣ Run LLM Response Generator

python src/llm_response.py

# 🛠️ Future Enhancements


🔹 Improve text cleaning for better retrieval

🔹 Support multi-PDF search

🔹 Optimize LLM prompt engineering


# Contributors 🏆

👤 Harsh Kumar – AI/ML Engineer
📧 Email: harshathghara19@gmail.com