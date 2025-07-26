import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import statistics
import json
import argparse
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
import time
import psutil
import os

# ============================================================================== 
# PHASE 1: PDF EXTRACTION & STRUCTURING
# ==============================================================================

def extract_and_classify_pages(pdf_path):
    doc = fitz.open(pdf_path)
    document_data = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            page_content = page.get_text("dict")
            content_type = "text"
        else:
            try:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                page_content = pytesseract.image_to_string(img, lang='eng')
                content_type = "image"
            except Exception as e:
                page_content, content_type = "", "error"
        document_data.append({
            "page_number": page_num + 1,
            "content_type": content_type,
            "content": page_content
        })
    doc.close()
    return document_data

def structure_document_from_text_pages(pages_data, source_document_name):
    sections = []
    all_font_sizes = [round(span["size"]) for page in pages_data if page["content_type"] == "text" for block in page["content"]["blocks"] if "lines" in block for line in block["lines"] for span in line["spans"]]
    if not all_font_sizes: return []
    body_text_size = statistics.mode(all_font_sizes)
    
    current_section = {"title": "Introduction", "level": 0, "content": "", "source_document": source_document_name, "page_number": 1}
    for page in pages_data:
        current_section["page_number"] = page["page_number"]
        if page["content_type"] == "text":
            for block in page["content"]["blocks"]:
                if "lines" in block and len(block["lines"]) == 1:
                    span = block["lines"][0]["spans"][0]
                    font_size, text = round(span["size"]), span["text"].strip()
                    if font_size > body_text_size and text:
                        if current_section["content"].strip(): sections.append(current_section)
                        level = 1 if font_size > body_text_size + 2 else 2
                        current_section = {"title": text, "level": level, "content": "", "source_document": source_document_name, "page_number": page["page_number"]}
                        continue
            # Replace the incorrect fitz.recover_text usage with direct extraction from spans
            plain_text = ""
            for block in page["content"]["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            plain_text += span["text"] + " "
            current_section["content"] += plain_text.strip() + "\n"
        elif page["content_type"] == "image":
            current_section["content"] += page["content"] + "\n"
    if current_section["content"].strip(): sections.append(current_section)
    return sections

# ============================================================================== 
# PHASE 2: DOMAIN CLASSIFICATION & SEMANTIC RETRIEVAL
# ==============================================================================

def classify_domain(document_text):
    domain_keywords = {
        "FINANCE": ['revenue', 'profit', 'investment', 'financials', 'assets', 'earnings', 'fiscal'],
        "ACADEMIC": ['abstract', 'methodology', 'dataset', 'results', 'conclusion', 'research', 'validation'],
        "BUSINESS": ['sales', 'marketing', 'customer', 'competitor', 'strategy', 'partnership', 'b2b'],
        "NEWS": ['report', 'source', 'interview', 'article', 'breaking', 'headline', 'journalist']
    }
    scores = {domain: sum(document_text.lower().count(kw) for kw in keywords) for domain, keywords in domain_keywords.items()}
    max_score = max(scores.values())
    return max(scores, key=scores.get) if max_score >= 10 else "GENERAL"

def retrieve_and_rank_sections(persona, job_to_be_done, document_sections, model):
    query = f"Persona: {persona}. Task: {job_to_be_done}"
    sections_with_content = [s for s in document_sections if s['content'].strip()]
    if not sections_with_content: return []
    section_texts = [f"{s['title']}\n{s['content']}" for s in sections_with_content]
    query_embedding = model.encode(query, convert_to_tensor=True)
    section_embeddings = model.encode(section_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, section_embeddings)
    for i, section in enumerate(sections_with_content):
        section['relevance_score'] = cosine_scores[0][i].item()
    return sorted(sections_with_content, key=lambda x: x['relevance_score'], reverse=True)

# ============================================================================== 
# PHASE 3: ADAPTIVE ANALYSIS & INSIGHT GENERATION
# ==============================================================================

def load_generative_model(domain):
    if domain == "FINANCE":
        model_id = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    else:
        model_id = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    print(f"Loaded model for '{domain}' domain: {model_id}")
    return model, tokenizer

def create_dynamic_prompt(persona, job_to_be_done, section_title, section_content):
    return f"**Your Role:** You are an AI assistant acting as a {persona}. **Your Task:** {job_to_be_done}. **Context:** Analyze the section titled '{section_title}'. **Instructions:** Based only on the text below, generate a concise and relevant analysis. **Text to Analyze:** {section_content}"

def generate_refined_text(model, tokenizer, domain, prompt, section_content):
    try:
        if domain == "FINANCE":
            question = prompt.split("Your Task: ")[1].split(". **Context")[0]
            inputs = tokenizer(question, section_content, return_tensors="pt", max_length=512, truncation=True)
            output = model(**inputs)
            answer_tokens = inputs["input_ids"][0, output.start_logits.argmax() : output.end_logits.argmax() + 1]
            refined_text = tokenizer.decode(answer_tokens)
            return refined_text if refined_text and "[CLS]" not in refined_text else "Could not extract a specific answer."
        else:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, num_beams=4, early_stopping=True)
            return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error during generation: {e}"

# ============================================================================== 
# PHASE 4: JSON ASSEMBLY & FINAL OUTPUT
# ==============================================================================

def assemble_final_json(input_docs, persona, job, ranked_sections):
    output = {
        "metadata": {
            "input_documents": input_docs,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [],
        "sub_section_analysis": []
    }
    for i, section in enumerate(ranked_sections):
        output["extracted_sections"].append({
            "Document": section["source_document"],
            "Page number": section["page_number"],
            "Section title": section["title"],
            "Importance_rank": i + 1
        })
        if "refined_text" in section:
            output["sub_section_analysis"].append({
                "Document": section["source_document"],
                "Page Number": section["page_number"],
                "Refined Text": section["refined_text"]
            })
    return json.dumps(output, indent=4, ensure_ascii=False)

# ============================================================================== 
# MAIN ORCHESTRATOR
# ==============================================================================

if __name__ == '__main__':
    # --- Start Profiling ---
    start_time = time.time()
    process = psutil.Process(os.getpid())
    peak_memory_usage = 0

    parser = argparse.ArgumentParser(description="Persona-Driven Document Intelligence Pipeline")
    parser.add_argument("-d", "--documents", nargs='+', required=True, help="List of paths to PDF documents.")
    parser.add_argument("-p", "--persona", type=str, required=True, help="The user persona (e.g., 'Investment Analyst').")
    parser.add_argument("-j", "--job", type=str, required=True, help="The job to be done.")
    args = parser.parse_args()

    print("--- Starting Pipeline ---")

    # 1. Load embedding model (once)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # 2. Process all documents
    all_sections = []
    for doc_path in args.documents:
        print(f"Processing document: {doc_path}")
        extracted_data = extract_and_classify_pages(doc_path)
        # Pass the document path to track the source
        document_sections = structure_document_from_text_pages(extracted_data, doc_path)
        all_sections.extend(document_sections)

    # 3. Classify domain and load appropriate generative model
    full_text = " ".join([s['content'] for s in all_sections])
    domain = classify_domain(full_text)
    generative_model, tokenizer = load_generative_model(domain)

    # 4. Rank sections based on relevance
    ranked_sections = retrieve_and_rank_sections(args.persona, args.job, all_sections, embedding_model)

    # 5. Generate insights for the top N sections
    print(f"\nGenerating insights for the top 5 relevant sections...")
    for section in ranked_sections[:5]: # Process top 5 for efficiency
        prompt = create_dynamic_prompt(args.persona, args.job, section['title'], section['content'])
        refined_text = generate_refined_text(generative_model, tokenizer, domain, prompt, section['content'])
        section['refined_text'] = refined_text

    # 6. Assemble and save the final JSON output
    final_json_output = assemble_final_json(args.documents, args.persona, args.job, ranked_sections)
    
    with open("output.json", "w", encoding="utf-8") as f:
        f.write(final_json_output)

    # --- Stop Profiling & Report Results ---
    end_time = time.time()
    total_time = end_time - start_time
    final_memory_mb = process.memory_info().rss / (1024 * 1024)

    print("\n--- Performance Metrics ---")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Final Memory Usage: {final_memory_mb:.2f} MB")
    print("--------------------------")

    print("\n--- Pipeline Finished ---")
    print("Final output saved to output.json") 