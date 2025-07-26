# Persona-Driven Document Intelligence Engine

This project is a submission for the Adobe India Hackathon 2025. It is a sophisticated Python pipeline that ingests a collection of PDF documents, understands a user's role (persona) and task, and generates targeted, AI-driven insights.

The system uses an adaptive approach, first classifying the document collection's domain (e.g., Finance, Academic) and then loading a specialized model to provide the most relevant analysis, all while adhering to strict performance and resource constraints.

---

## Features

-   **Adaptive PDF Parsing:** Intelligently handles both text-based and scanned (image-based) PDFs using OCR.
-   **Hierarchical Sectioning:** Automatically identifies and structures document content based on heading styles.
-   **Domain Classification:** A lightweight, keyword-based classifier to identify the document's subject matter.
-   **Semantic Search:** Uses state-of-the-art sentence-transformer models to rank document sections by relevance to a user's query.
-   **Persona-Driven Generation:** Employs specialized generative models (like FinBERT for finance or a quantized DistilBART for general summaries) to produce high-quality, targeted insights.
-   **Constraint-Aware:** Engineered to run offline on a CPU, within a 1GB memory budget and a 60-second time limit.

---

## Setup & Installation

**Prerequisite:** You must have the **Tesseract OCR engine** installed on your system.
-   **Windows:** Download and install from the [official Tesseract wiki](https://github.com/UB-Mannheim/tesseract/wiki). Ensure you add it to your system's PATH during installation.
-   **macOS:** `brew install tesseract`
-   **Linux (Ubuntu/Debian):** `sudo apt-get install tesseract-ocr`

**Instructions:**

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <repo-name>
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv env
    source env/bin/activate  # On macOS/Linux
    # .\env\Scripts\activate   # On Windows
    ```

3.  **Install all required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *This will install PyMuPDF, Transformers, PyTorch, Sentence-Transformers, etc. The first time you run the main script, the necessary AI models will be downloaded and cached automatically.*

---

## How to Run

The application is run from the command line. You must provide the paths to the documents, the persona, and the job to be done.

### Example Usage:

```bash
python main.py \
  --documents path/to/report1.pdf path/to/research_paper.pdf \
  --persona "Investment Analyst" \
  --job "Analyze revenue trends, R&D investments, and market positioning strategies"
```

The script will process the documents and generate a detailed output.json file in the root directory.