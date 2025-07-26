import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import statistics
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering

# --- IMPORTANT ---
# On your first run, you might need to tell pytesseract where to find the Tesseract installation.
# Uncomment the line below and set the path if you get a "TesseractNotFoundError".
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example for Windows

def extract_and_classify_pages(pdf_path):
    """
    Opens a PDF and extracts content from each page, classifying them as text or image.

    Args:
        pdf_path (str): The file path to the PDF.

    Returns:
        list: A list of page data dictionaries. Each dict contains page number,
              content type ('text' or 'image'), and the extracted content.
    """
    doc = fitz.open(pdf_path)
    document_data = []
    print(f"Processing {doc.page_count} pages...")

    for page_num, page in enumerate(doc):
        page_content = ""
        content_type = ""

        # Attempt to extract text directly
        text = page.get_text()

        if text.strip():
            # If text is found, it's a text-based page
            content_type = "text"
            # Use the 'dict' option to get detailed structure with font info
            page_content = page.get_text("dict")
            print(f"  Page {page_num + 1}: Classified as TEXT.")
        else:
            # If no text, it's an image-based page needing OCR
            content_type = "image"
            print(f"  Page {page_num + 1}: Classified as IMAGE. Performing OCR...")
            try:
                # Render page to an image
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))

                # Use Tesseract to extract text from the image
                page_content = pytesseract.image_to_string(img, lang='eng')
            except Exception as e:
                print(f"    ERROR: OCR failed on page {page_num + 1}. Reason: {e}")
                page_content = "" # Store empty content if OCR fails

        document_data.append({
            "page_number": page_num + 1,
            "content_type": content_type,
            "content": page_content
        })

    doc.close()
    return document_data

def structure_document_from_text_pages(pages_data):
    """
    Identifies headings and structures content from text-based pages.

    Args:
        pages_data (list): The output from extract_and_classify_pages.

    Returns:
        list: A list of structured sections, each with a title, level, and content.
    """
    sections = []
    all_font_sizes = []

    # --- Part 1: Gather font size statistics from all text pages ---
    for page in pages_data:
        if page["content_type"] == "text":
            for block in page["content"]["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            all_font_sizes.append(round(span["size"]))

    if not all_font_sizes:
        print("No text-based pages found to structure.")
        return []

    # Determine the most common font size (body text)
    body_text_size = statistics.mode(all_font_sizes)
    print(f"\nIdentified body text font size: {body_text_size}")

    # --- Part 2: Identify headings and create sections ---
    current_section = {"title": "Introduction", "level": 0, "content": ""}
    for page in pages_data:
        if page["content_type"] == "text":
            for block in page["content"]["blocks"]:
                if "lines" in block:
                    # Heuristic: A block is a potential heading if it has one line
                    # and its font size is larger than the body text.
                    if len(block["lines"]) == 1:
                        line = block["lines"][0]
                        span = line["spans"][0]
                        font_size = round(span["size"])
                        text = span["text"].strip()

                        if font_size > body_text_size and text:
                            # We found a heading. Save the previous section.
                            if current_section["content"].strip():
                                sections.append(current_section)

                            # Start a new section
                            heading_level = 1 if font_size > body_text_size + 2 else 2 # Simple H1/H2 logic
                            current_section = {
                                "title": text,
                                "level": heading_level,
                                "content": ""
                            }
                            print(f"  Found H{heading_level} Heading: '{text}'")
                            continue # Move to the next block

            # If not a heading, append its text content to the current section
            # For simplicity, concatenate all text spans
            for block in page["content"]["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            current_section["content"] += span["text"] + " "
            current_section["content"] += "\n"

        elif page["content_type"] == "image":
            # For image pages, append the OCR'd content to the current section
            current_section["content"] += page["content"] + "\n"

    # Add the last processed section
    if current_section["content"].strip():
        sections.append(current_section)

    return sections

def classify_domain(document_text):
    """
    Classifies the document collection's domain based on keyword frequency
    to handle diverse personas like Researchers, Journalists, and Salespeople.

    Args:
        document_text (str): The concatenated text of all documents.

    Returns:
        str: The classified domain ('FINANCE', 'ACADEMIC', 'BUSINESS', 'NEWS', or 'GENERAL').
    """
    # Expanded keyword sets to cover more domains and personas
    domain_keywords = {
        "FINANCE": [
            'revenue', 'profit', 'loss', 'investment', 'equity', 'shares',
            'market', 'financials', 'assets', 'liabilities', 'ipo', 'earnings',
            'quarter', 'fiscal', 'balance sheet', 'cash flow', 'valuation'
        ],
        "ACADEMIC": [
            'abstract', 'introduction', 'methodology', 'dataset', 'results',
            'conclusion', 'experiment', 'hypothesis', 'study', 'journal',
            'publication', 'research', 'analysis', 'validation', 'literature review',
            'citation', 'appendix'
        ],
        "BUSINESS": [
            'sales', 'marketing', 'customer', 'competitor', 'growth', 'strategy',
            'partnership', 'b2b', 'b2c', 'pitch', 'startup', 'venture', 'swot',
            'business plan', 'stakeholder'
        ],
        "NEWS": [
            'report', 'source', 'interview', 'article', 'breaking', 'investigation',
            'headline', 'journalist', 'press', 'coverage', 'column', 'front page',
            'current events'
        ]
    }

    scores = {domain: 0 for domain in domain_keywords}
    
    # Convert text to lowercase for case-insensitive matching
    lower_text = document_text.lower()

    # Score each domain by counting keyword occurrences
    for domain, keywords in domain_keywords.items():
        for keyword in keywords:
            scores[domain] += lower_text.count(keyword)

    # Determine the winning domain
    max_score = 0
    classified_domain = "GENERAL"
    for domain, score in scores.items():
        if score > max_score:
            max_score = score
            classified_domain = domain

    # A simple threshold to avoid misclassification on documents with few keywords
    # Increased threshold slightly for more robust classification
    if max_score < 10:
        classified_domain = "GENERAL"
        
    print(f"\nDomain classification scores: {scores}")
    print(f"==> Classified domain as: {classified_domain}")
    
    return classified_domain

def load_generative_model(domain):
    """
    Loads the appropriate generative model and tokenizer based on the classified domain.

    Args:
        domain (str): The domain tag ('FINANCE', 'ACADEMIC', 'GENERAL', etc.).

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    model_id = ""
    print(f"\n--- Adaptive Model Loading ---")
    print(f"Domain is '{domain}'. Loading appropriate model...")

    if domain == "FINANCE":
        # FinBERT is specialized for financial text. It's best for extractive tasks.
        # For a hackathon, we can frame its output as a "generative insight".
        # Note: This is a different architecture, so we load it differently.
        model_id = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # We load FinBERT as a Question-Answering model for this example
        model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        print(f"Loaded specialized model: {model_id}")

    # For all other cases, we use a general-purpose, lightweight T5 model.
    else:
        model_id = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        print(f"Loaded general-purpose model: {model_id}")

    return model, tokenizer

def retrieve_and_rank_sections(persona, job_to_be_done, document_sections, model):
    """
    Ranks document sections based on their relevance to a user's query.

    Args:
        persona (str): The user's role (e.g., "Investment Analyst").
        job_to_be_done (str): The user's specific task.
        document_sections (list): The list of structured sections from the documents.
        model (SentenceTransformer): The pre-loaded sentence-transformer model.

    Returns:
        list: The list of document sections, sorted by relevance, with a new
              'relevance_score' key added to each section.
    """
    # a. Ingest persona + job-to-be-done to form a query
    # This creates a rich, descriptive query that captures the user's full intent.
    query = f"Persona: {persona}. Task: {job_to_be_done}"
    print(f"\nGenerated Query for Semantic Search: '{query}'")

    # For efficiency, only embed sections that have meaningful content.
    sections_with_content = [s for s in document_sections if s['content'].strip()]
    if not sections_with_content:
        print("No content found in sections to rank.")
        return []
        
    # Extract the text from each section to be embedded
    section_texts = [f"{s['title']}\n{s['content']}" for s in sections_with_content]

    # b. Embed the query and all sections using the model
    print("Embedding query and document sections...")
    query_embedding = model.encode(query, convert_to_tensor=True)
    section_embeddings = model.encode(section_texts, convert_to_tensor=True)
    print(f"Embeddings generated for {len(section_embeddings)} sections.")

    # c. Rank sections using cosine similarity
    # This computes the similarity between the single query and all section embeddings
    cosine_scores = util.cos_sim(query_embedding, section_embeddings)

    # Add the relevance score to each section dictionary
    for i, section in enumerate(sections_with_content):
        section['relevance_score'] = cosine_scores[0][i].item()

    # Sort the sections in descending order based on the score
    ranked_sections = sorted(sections_with_content, key=lambda x: x['relevance_score'], reverse=True)

    return ranked_sections

def create_dynamic_prompt(persona, job_to_be_done, section_title, section_content):
    """
    Constructs a detailed, instruction-based prompt for the generative model.

    Args:
        persona (str): The user's role.
        job_to_be_done (str): The user's specific task.
        section_title (str): The title of the document section.
        section_content (str): The text content of the document section.

    Returns:
        str: A formatted, detailed prompt ready for the model.
    """
    prompt = f"""
    **Your Role:** You are a helpful AI assistant acting as a {persona}.

    **Your Task:** Your goal is to {job_to_be_done}.

    **Context:** You are analyzing the following section titled \"{section_title}\".

    **Instructions:** Based *only* on the text provided below, generate a concise and relevant analysis that directly addresses the task. Do not invent information.

    ---
    **Text to Analyze:**
    {section_content}
    ---
    """
    return prompt

def generate_refined_text(model, tokenizer, domain, prompt, section_content):
    """
    Generates a concise, persona-driven insight using the appropriate model.

    Args:
        model: The pre-loaded generative model.
        tokenizer: The corresponding tokenizer.
        domain (str): The classified domain ('FINANCE', 'GENERAL', etc.).
        prompt (str): The dynamically generated prompt for the model.
        section_content (str): The raw text content of the section.

    Returns:
        str: The generated "Refined Text" insight.
    """
    # The generation process is different for different model architectures.
    if domain == "FINANCE":
        # For FinBERT (a Question-Answering model), we treat the 'job_to_be_done'
        # from the prompt as the 'question' and the section as the 'context'.
        try:
            # Extract the task to use as the question for the QA model
            question = prompt.split("Your Task: Your goal is to ")[1].split("\n")[0]
            inputs = tokenizer(question, section_content, return_tensors="pt", max_length=512, truncation=True)
            
            output = model(**inputs)
            start_index = output.start_logits.argmax()
            end_index = output.end_logits.argmax()
            
            answer_tokens = inputs["input_ids"][0, start_index : end_index + 1]
            refined_text = tokenizer.decode(answer_tokens)
            
            if not refined_text or "[CLS]" in refined_text:
                return "Could not extract a specific answer. The section may not contain the requested financial details."
                
            return refined_text
                
        except Exception as e:
            # Fallback for any errors during extractive QA
            return f"Error during financial analysis: {e}"
            
    else:
        # For t5-small (a sequence-to-sequence model), the process is straightforward.
        try:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(
                inputs.input_ids,
                max_length=150,  # Set a reasonable max length for the summary
                min_length=40,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            refined_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return refined_text
            
        except Exception as e:
            return f"Error during text generation: {e}"

if __name__ == '__main__':
    # --- Configuration ---
    pdf_file_path = "sample/editorial_generator_proposal_final.pdf"
    user_persona = "Academic Researcher"
    user_job = "Summarize the methodology and key findings of the study"

    # --- Model Loading (Embedding Model) ---
    print("Loading the sentence-transformer model (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully.")

    # --- Execution Pipeline ---
    # 1. & 2. Extract, Classify, and Structure
    extracted_data = extract_and_classify_pages(pdf_file_path)
    document_sections = structure_document_from_text_pages(extracted_data)

    # 3. Classify Domain
    full_text_for_classification = " ".join([s['content'] for s in document_sections])
    domain = classify_domain(full_text_for_classification)

    # 4. Adaptive Model Loading (NEW STEP)
    # This loads the correct generative model based on the domain.
    generative_model, tokenizer = load_generative_model(domain)

    # 5. Retrieve and Rank Relevant Sections
    ranked_document_sections = retrieve_and_rank_sections(
        user_persona,
        user_job,
        document_sections,
        embedding_model
    )

    # --- Preview the Results ---
    print("\n--- Top 5 Most Relevant Sections ---")
    for i, section in enumerate(ranked_document_sections[:5]):
        print(f"\nRank {i+1}: {section['title']} (Relevance: {section['relevance_score']:.4f})")
        # In the next step, we will use the loaded generative_model to process this content.
        print(f"  Content: {section['content'][:150].strip()}...")

    # --- Preview the Analysis Prompts ---
    print("\n--- Dynamically Generated Prompts for Top 3 Sections ---")
    for i, section in enumerate(ranked_document_sections[:3]):
        analysis_prompt = create_dynamic_prompt(
            user_persona,
            user_job,
            section['title'],
            section['content']
        )
        print(f"\n--- Prompt for Rank {i+1} Section: '{section['title']}' ---")
        print(analysis_prompt)
        # In the next phase, this 'analysis_prompt' will be sent to the generative_model.

    # --- Final Analysis and Output ---
    print("\n--- Generating Final Insights for Top 3 Sections ---")
    for i, section in enumerate(ranked_document_sections[:3]):
        analysis_prompt = create_dynamic_prompt(
            user_persona,
            user_job,
            section['title'],
            section['content']
        )
        print(f"\nProcessing Rank {i+1} Section: '{section['title']}'...")
        refined_text = generate_refined_text(
            generative_model,
            tokenizer,
            domain,
            analysis_prompt,
            section['content']
        )
        section['refined_text'] = refined_text
        print(f"  Relevance Score: {section['relevance_score']:.4f}")
        print(f"  **Refined Text Insight:** {section['refined_text']}") 