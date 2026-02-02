import os
import json
from tqdm import tqdm

EXTRACTED_DATA_FOLDER = "extracted_data"
INDEX_FILE = "indexed_legal_data.json"

def create_search_index():
    """Reads all JSON files from the extracted_data folder and creates a single, indexed JSON file."""
    
    # Path to the indexed file in the root 'data' directory
    output_path = os.path.join("data", INDEX_FILE)
    
    # 1. Check for extracted files
    extracted_files = [f for f in os.listdir(EXTRACTED_DATA_FOLDER) if f.endswith(".json")]
    
    if not extracted_files:
        print(f"FATAL ERROR: No JSON files found in '{EXTRACTED_DATA_FOLDER}'. Run pdf_extraction.py first.")
        return

    print(f"INFO: Found {len(extracted_files)} data files to index.")
    
    master_index = []
    document_id = 1

    # 2. Process each extracted JSON file
    for filename in tqdm(extracted_files, desc="Indexing Legal Documents"):
        file_path = os.path.join(EXTRACTED_DATA_FOLDER, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                tqdm.write(f"WARNING: File '{filename}' is empty. Skipping.")
                continue

            # 3. Flatten the data into a single searchable index
            for section in data:
                # Add a unique index ID and a clear document name (e.g., 'IPC-1860')
                source_name = filename.replace(".json", "").replace("THE-", "").replace("-", " ").title().replace(" ", "-")

                master_index.append({
                    "id": document_id,
                    "document_name": source_name,
                    "page_number": section.get("page_number", 0),
                    "raw_text": section.get("text", "").strip(),
                    # Future fields like category or section number can be added here
                })
                document_id += 1

        except json.JSONDecodeError:
            tqdm.write(f"❌ ERROR: Failed to parse JSON in '{filename}'. Skipping.")
        except Exception as e:
            tqdm.write(f"❌ UNEXPECTED ERROR processing '{filename}': {e}. Skipping.")

    # 4. Save the Master Index
    if master_index:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(master_index, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Successfully created master index with {len(master_index)} entries.")
        print(f"Index saved to: {output_path}")
    else:
        print("\n⚠️ SKIPPED: Master index is empty. No usable data was processed.")


if __name__ == "__main__":
    create_search_index()