# RAG Complaint Chatbot using CFPB Consumer Complaints Dataset

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions about consumer financial complaints and retrieve relevant complaint information using semantic search.

The system processes the Consumer Financial Protection Bureau (CFPB) complaint dataset, transforms complaint narratives into searchable vector representations, and uses ChromaDB as a vector database for efficient similarity-based retrieval.

The goal is to build an AI-powered complaint analysis assistant capable of finding relevant customer experiences and supporting better understanding of financial service issues.

---

# Architecture Overview

The project follows an end-to-end RAG pipeline:

```
Raw CFPB Dataset
        |
        v
Data Cleaning & Preprocessing
        |
        v
Stratified Sampling
        |
        v
Text Chunking
        |
        v
Embedding Generation
        |
        v
ChromaDB Vector Store
        |
        v
Semantic Retrieval
        |
        v
RAG Response Generation
```

---

# Features

## Data Processing

* Loads and processes CFPB consumer complaint data
* Cleans complaint narratives
* Creates structured product categories
* Removes unusable or empty complaint records

## Intelligent Sampling

A stratified sampling strategy is applied to preserve the original distribution of complaint categories.

Sample size:

* 12,000 complaints

Categories:

* Credit Card
* Savings Account
* Money Transfer
* Personal Loan

---

## Text Chunking

Long complaint narratives are split into smaller semantic chunks using:

* RecursiveCharacterTextSplitter

Configuration:

```
chunk_size = 500
chunk_overlap = 50
```

This improves retrieval accuracy while maintaining contextual information.

---

## Embedding Model

The project uses:

```
sentence-transformers/all-MiniLM-L6-v2
```

Reason for selection:

* Efficient inference speed
* Strong semantic similarity performance
* Compact 384-dimensional embeddings
* Suitable for large-scale retrieval tasks

---

## Vector Database

Vector storage is implemented using:

```
ChromaDB
```

Each chunk stores metadata:

```json
{
 "complaint_id": "...",
 "product_category": "..."
}
```

This allows retrieved information to be traced back to the original complaint.

---

# Project Structure

```
rag-complaint-chatbot/

│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── task1_eda.ipynb
│   └── task2_vector_indexing.ipynb
│
├── src/
│   ├── data_processor.py
│   ├── run_preprocessing.py
│   ├── sampling.py
│   ├── chunking.py
│   ├── vector_store.py
│   └── build_vector_db.py
│
├── vector_store/
│
├── requirements.txt
│
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone <repository-url>

cd rag-complaint-chatbot
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

## Step 1: Preprocess Dataset

```bash
python src/run_preprocessing.py
```

This generates the cleaned complaint dataset.

---

## Step 2: Build Vector Database

```bash
python src/build_vector_db.py
```

This performs:

* Sampling
* Document creation
* Chunking
* Embedding generation
* ChromaDB indexing

---

# Validation Results

Current Task 2 results:

| Metric                       | Value   |
| ---------------------------- | ------- |
| Original Complaints          | 445,456 |
| Sampled Complaints           | 12,000  |
| Generated Chunks             | 34,079  |
| Average Chunks per Complaint | 2.84    |
| Embedding Dimension          | 384     |

---

# Error Handling and Reliability

The project includes defensive programming practices:

* File existence validation
* Empty dataset checks
* Configuration validation
* Exception handling around pipeline steps
* Logging for easier debugging

Example:

```
Dataset Loading
      |
      v
Validation Check
      |
      v
Processing
      |
      v
Error Logging if Failure Occurs
```

---

# Future Improvements

Possible extensions:

* Integrate an LLM for answer generation
* Add Streamlit chatbot interface
* Implement retrieval evaluation metrics
* Add conversation memory
* Deploy as a cloud application

---

# Technologies Used

Python

Pandas

LangChain

Sentence Transformers

ChromaDB

Streamlit

Machine Learning

Natural Language Processing

Retrieval-Augmented Generation (RAG)

---

# Author

Built as an AI engineering project exploring document retrieval, embeddings, and RAG application development.
