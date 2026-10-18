# IdeaGen Prototype Demo Guide

This document outlines the steps to run the IdeaGen Real-Data Proof-of-Concept Demonstration locally.

**Important Notice**: This is a real-data prototype demonstration using a bounded number of research papers. It is not a full-corpus literature search engine, and we do not claim comprehensive literature coverage. The large-scale OpenAlex acquisition is deferred.

## Prerequisites
- Node.js installed for the frontend
- Python 3.11 installed for the backend
- Docker (optional) for the database, or in-memory SQLite (default for demo)

## 1. Start IdeaGen Backend
Open a terminal in the `backend` directory and start the FastAPI dev server:
```bash
cd backend
python -m uvicorn ideagen.api.main:app --reload --port 8000
```
*(The backend loads an in-memory SQLite database populated strictly by our prototype corpus defined in `docs/prototype_corpus_v1.json`)*

## 2. Start IdeaGen Frontend
Open a separate terminal in the `frontend` directory and start the Vite dev server:
```bash
cd frontend
npm install
npm run dev
```

## 3. Open the Dashboard
Navigate to the local URL provided by Vite (e.g. `http://localhost:5173`).

## Presentation Flow

The dashboard allows you to start the presentation in two distinct ways:

### PATH A: Interactive Live Search

Follow this path to demonstrate dynamic corpus assembly via live APIs.

1. **Search a topic**: Click **Search Papers** on the home screen. Enter a topic like `"hallucination in retrieval augmented generation"`.
2. **Show papers**: The system will dynamically fetch up to 15 real papers from the arXiv API. Point out that these are real papers (real titles, authors, dates, abstracts).
3. **Analyze**: Select the papers you wish to analyze, then click **Analyze Papers**.
4. **Show opportunity**: After the progress screen completes, click **View Opportunities**. Point to the generated opportunity candidate and the problem statement constructed from recurring gaps.
5. **Show evidence**: Click the opportunity. Scroll down to show the **Supporting Evidence** excerpts extracted directly from the real original abstracts.
6. **Show evolution**: Scroll to the **Evolution** section. Explain how IdeaGen categorized the retrieved papers chronologically (e.g., Baseline vs Evolution batches) and how the opportunity status transitions.

### PATH B: Manual PDF Upload

Follow this path to demonstrate bringing your own research.

1. **Upload research papers**: Click **Upload PDFs** on the home screen. Drag and drop 3-5 real research paper PDFs into the dropzone.
2. **Show uploaded papers**: The system extracts the text. Show the list of successfully processed papers ready for review.
3. **Analyze**: Select the papers you wish to analyze, then click **Analyze Papers**.
4. **Show opportunity**: After the progress screen completes, click **View Opportunities**. Point to the generated opportunity candidate.
5. **Show evidence**: Click the opportunity. Scroll down to show the **Supporting Evidence** excerpts extracted directly from your uploaded PDFs.
6. **Show evolution**: Scroll to the **Evolution** section. Show how the chronological relationships between your uploaded papers deterministically transition the opportunity state.

## Fallback Mode
If live paper search is temporarily unavailable (e.g., network issues or arXiv API limits), IdeaGen provides a resilient fallback:
- An error will be displayed indicating live search is unavailable.
- Click the **Use Demo Corpus** button.
- This immediately loads the pre-packaged 20-paper curated real-data corpus and executes the identical pipeline, ensuring the presentation can continue flawlessly.
