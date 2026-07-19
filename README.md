# Movie Recommender

An interactive, full-stack web application that delivers personalized film suggestions. Powered by a natural language processing (NLP) pipeline, the system builds text-based feature matrices from film metadata to compute directional similarity profiles, serving recommendations instantly through a lightweight Flask web dashboard.


## Features

*   **Content-Based Filtering Engine:** Constructs a metadata "soup" combining processed textual markers (keywords, cast, director, and genres) to discover underlying content patterns.
*   **Vectorized Tokenization Pipeline:** Utilizes `CountVectorizer` to remove stop words and transform clean string variables into structural numerical token counts.
*   **Cosine Similarity Matching:** Evaluates the cosine distance across vector dimensions to isolate and rank the top $10$ most structurally relevant cinematic matches for any given title.
*   **Dynamic Auto-Fill API:** Features an asynchronous `/autocomplete` endpoint allowing frontend widgets to serve real-time search engine suggestions as users type.
*   **Double-Dataset Integration:** Merges granular credit summaries (`tmdb_5000_credits.csv`) alongside core metadata collections (`tmdb_5000_movies.csv`) to enrich feature profiles.


## Architecture & Project Structure

The codebase partitions structural file streams, visualization assets, and server routing routines into distinct layers:

```text
├── static/                  # Client-side style layouts and UI assets
├── templates/               # Presentation View Models
│   ├── index.html           # Initial landing view and search interface
│   └── recommendations.html # Results page presenting generated output arrays
│
├── tmdb_5000_movies.csv     # Primary dataset containing budget, genres, and metadata fields
├── tmdb_5000_credits.csv    # Supplementary dataset tracking cast and crew details
├── app.py                   # Consolidated web router, text processor, and similarity engine
│
├── Project Report.docx      # Technical architecture write-up
├── Working of The Program.docx # Engine logic flowchart documentation
└── README.md                # System documentation manual
```

## How It Works

1. **Data Ingestion & Merging:** The engine reads both target CSV tables using `pandas` and merges them via unique identification indices, safely normalizing conflicting column headers.

2. **Data Sanitization & Soup Creation:** Categorical features are parsed out from stringified JSON representations using literal evaluation rules. Spaces are stripped and text is shifted to lowercase to ensure multi-word entities (like director names `Christopher Nolan` vs. `Christopher Miller`) vectorize as individual distinct tokens (`christophernolan`).

3. **Similarity Assessment:** A sparse count matrix is fit across text representations. The system computes a dense similarity matrix using the formula:

   $$\text{Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

4. **Endpoint Fulfillment:** When a client issues a POST query matching a title index, `app.py` cross-references the precomputed indices array, isolates matching score metrics, and serves the top 10 relevant records.

## Quick Start

### 1. Prerequisites

Ensure you have **Python 3.x** installed along with the essential dependencies:

Bash

```
pip install flask pandas numpy scikit-learn
```

### 2. File Placement

Make sure `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` are placed directly in the project root directory alongside `app.py`.

### 3. Execution

Launch the Flask development server:

Bash

```
python app.py
```

> Open your browser and navigate to `http://127.0.0.1:5000/` to test your search and recommendation pipelines.
