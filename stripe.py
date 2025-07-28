import os
import fitz  # PyMuPDF
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_FOLDERS = [
    os.path.join(BASE_DIR, "Collection 1", "PDFS"),
    os.path.join(BASE_DIR, "Collection 2", "PDFS"),
    os.path.join(BASE_DIR, "Collection 3", "PDFS")
]

OUTPUT_FILE = os.path.join(BASE_DIR, "output", "persona_output.json")

# Customize your search here
PERSONA = "HR Manager"
TASK = "Create onboarding material"

def extract_text_chunks(pdf_path):
    chunks = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                chunks.append({
                    "document": os.path.basename(pdf_path),
                    "page": page_num,
                    "text": text
                })
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return chunks

def rank_chunks(chunks, persona, task):
    corpus = [chunk["text"] for chunk in chunks]
    query = [f"{persona} {task}"]
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(query + corpus)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    for i, score in enumerate(scores):
        chunks[i]["score"] = float(score)
    return sorted(chunks, key=lambda x: x["score"], reverse=True)[:10]

def main():
    all_chunks = []
    print("Looking for PDFs...")

    for folder in PDF_FOLDERS:
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        for file in os.listdir(folder):
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(folder, file)
                print(f"Reading: {pdf_path}")
                chunks = extract_text_chunks(pdf_path)
                all_chunks.extend(chunks)

    if not all_chunks:
        print("No valid PDF text found. Exiting.")
        return

    print(f"Found {len(all_chunks)} total text chunks")

    top_sections = rank_chunks(all_chunks, PERSONA, TASK)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {"persona": PERSONA, "task": TASK},
            "sections": top_sections
        }, f, indent=2, ensure_ascii=False)

    print(f"Output written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
