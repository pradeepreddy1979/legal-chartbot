import re
import json
from PyPDF2 import PdfReader # Requires 'pip install PyPDF2'
import os
from tqdm import tqdm

# --- CONFIGURATION ---
# >>> UPDATE THIS LINE with your absolute path <<<
# Example for Windows: r"C:\Users\YourUsername\Downloads\bns.pdf"
# Example for macOS/Linux: "/Users/username/Downloads/bns.pdf"
PDF_FILE_PATH = "data/acts/bns.pdf"  # Update with actual PDF file path
OUTPUT_DIR = "extracted_data"
# Regex to robustly split the law by section number (e.g., "1. Short title...", "2. Definitions...")
SECTION_PATTERN = re.compile(r'\n\s*(\d+)\.\s*(.+?)(?=\n\s*\d+\.|\n\s*CHAPTER|\Z)', re.DOTALL)

def extract_text_from_pdf(pdf_path):
    """Extracts raw text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        # Loop through pages and extract text
        for page in tqdm(reader.pages, desc="Extracting text from PDF"):
            text += page.extract_text() + "\n\n"
        return text
    except FileNotFoundError:
        print(f"FATAL ERROR: PDF file not found at {pdf_path}. Check the path.")
        return None

def extract_and_save_legal_sections(raw_text, law_name="BNS"):
    """Splits raw legal text into sections and saves them in the required JSON format."""
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    extracted_sections = []
    
    # Find all section matches
    matches = list(SECTION_PATTERN.finditer(raw_text))

    if not matches:
        print("WARNING: No sections found using the defined pattern. Extraction failed.")
        return

    for match in tqdm(matches, desc="Structuring Sections"):
        section_number = int(match.group(1).strip())
        full_text_content = match.group(2).strip()

        # Clean the text (remove excessive whitespace)
        cleaned_text = re.sub(r'\s+', ' ', full_text_content).strip()

        # Extract title (heuristic: text until the first closing bracket or period)
        article_title_match = re.search(r'^(.*?)[.)]', cleaned_text)
        article_title = article_title_match.group(1).strip() if article_title_match else f"Section {section_number} Provision"

        extracted_sections.append({
            # The indexer uses 'document_name', 'page_number', and 'raw_text'
            "document_name": law_name,
            "page_number": 0, # Cannot reliably get page number from simple text extraction
            "text": cleaned_text, # This is the "raw_text" for your indexer
            "article_number": section_number,
            "article_title": article_title,
        })

    # Save to JSON file as expected by index_data.py
    source_name_clean = law_name.replace("THE-", "").replace("-", " ").title().replace(" ", "-")
    output_filename = os.path.join(OUTPUT_DIR, f"{source_name_clean}.json")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(extracted_sections, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Successfully extracted {len(extracted_sections)} sections.")
    print(f"File saved to: {output_filename}")


if __name__ == "__main__":
    bns_text = extract_text_from_pdf(PDF_FILE_PATH)
    
    if bns_text:
        # Pass a sanitized name for the filename
        extract_and_save_legal_sections(bns_text, "THE-BHARATIYA-NYAYA-SANHITA-2023")