import os
import fitz  # PyMuPDF
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Base path is where persona.py lives
BASE_DIR = os.path.dirname(os.path.abspath(_file_))

PDF_FOLDERS = [
    os.path.join(BASE_DIR, "Collection 1", "PDFS"),
    os.path.join(BASE_DIR, "Collection 2", "PDFS"),
    os.path.join(BASE_DIR, "Collection 3", "PDFS")
]

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Customize your use-case
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
        print(f"❌ Error reading {pdf_path}: {e}")
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

def process_pdf(pdf_path):
    chunks = extract_text_chunks(pdf_path)
    if not chunks:
        print(f"⚠ No text found in {pdf_path}")
        return None

    top_sections = rank_chunks(chunks, PERSONA, TASK)
    return {
        "metadata": {
            "persona": PERSONA,
            "task": TASK,
            "source": os.path.basename(pdf_path)
        },
        "sections": top_sections
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for folder in PDF_FOLDERS:
        if not os.path.exists(folder):
            print(f"⚠ Folder not found: {folder}")
            continue

        for file in os.listdir(folder):
            if not file.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(folder, file)
            print(f"📄 Processing: {pdf_path}")
            result = process_pdf(pdf_path)

            if result:
                base_name = os.path.splitext(file)[0]
                output_file = os.path.join(OUTPUT_DIR, f"{base_name}_output.json")

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"✅ Output saved to: {output_file}")

if _name_ == "_main_":
    main()
