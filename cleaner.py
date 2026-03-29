import os
import re
import time
from google import genai

# ============================================
# CONFIGURATION (Uses GitHub Secrets)
# ============================================
# This looks for the secret we will set up in GitHub
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-1.5-flash"  # Stable version for batch processing

# File Settings
INPUT_FILE = "input.txt"         # Put your OCR text in this file
OUTPUT_FOLDER = "cleaned_pages"
FINAL_OUTPUT = "cleaned_full.txt"

# Processing Settings
BATCH_SIZE = 5
DELAY_BETWEEN_REQUESTS = 10      # Increased slightly for stability in background

# ============================================
# LOGIC FUNCTIONS
# ============================================

def parse_pages(file_path):
    """Split the text file into pages based on --- Page X --- markers."""
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found!")
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r'(---\s*Page\s*\d+\s*---)'
    parts = re.split(pattern, content)

    pages = {}
    current_page_num = None

    for part in parts:
        header_match = re.match(r'---\s*Page\s*(\d+)\s*---', part.strip())
        if header_match:
            current_page_num = int(header_match.group(1))
        elif current_page_num is not None:
            pages[current_page_num] = part.strip()

    print(f"✅ Found {len(pages)} pages in the file.")
    return pages

def create_batches(pages, batch_size=5):
    sorted_page_nums = sorted(pages.keys())
    batches = []
    for i in range(0, len(sorted_page_nums), batch_size):
        batch_nums = sorted_page_nums[i:i + batch_size]
        batch = {num: pages[num] for num in batch_nums}
        batches.append(batch)
    return batches

def clean_batch_with_gemini(client, batch, batch_index, total_batches):
    pages_text = ""
    for page_num in sorted(batch.keys()):
        pages_text += f"\n--- Page {page_num} ---\n{batch[page_num]}\n"

    prompt = f"""أنت خبير في تصحيح النصوص العربية المستخرجة من OCR.
المطلوب:
1. صحح الأخطاء الإملائية والنحوية الناتجة عن OCR
2. أعد ترتيب النص بشكل منطقي ومقروء
3. أزل الأحرف والرموز الغريبة
4. حافظ على فواصل الصفحات بنفس التنسيق --- Page X ---
5. لا تضف محتوى جديد، فقط نظّف الموجود.

النص:
{pages_text}"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        print(f"  ✅ Batch {batch_index + 1}/{total_batches} Success.")
        return response.text
    except Exception as e:
        print(f"  ❌ Batch {batch_index + 1} FAILED: {str(e)}")
        return pages_text # Return original text if AI fails to prevent data loss

def process_all():
    if not GEMINI_API_KEY:
        print("❌ API Key missing! Check your GitHub Secrets.")
        return

    client = genai.Client(api_key=GEMINI_API_KEY)
    pages = parse_pages(INPUT_FILE)
    
    if not pages:
        return

    batches = create_batches(pages, BATCH_SIZE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    total_batches = len(batches)
    print(f"🚀 Starting background process for {total_batches} batches...")

    for i, batch in enumerate(batches):
        cleaned = clean_batch_with_gemini(client, batch, i, total_batches)
        
        # Save individual batch
        min_p, max_p = min(batch.keys()), max(batch.keys())
        batch_file = os.path.join(OUTPUT_FOLDER, f"batch_{i+1:03d}.txt")
        with open(batch_file, "w", encoding="utf-8") as f:
            f.write(cleaned)

        if i < total_batches - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    # Final Merge
    with open(FINAL_OUTPUT, "w", encoding="utf-8") as outfile:
        batch_files = sorted(os.listdir(OUTPUT_FOLDER))
        for bf in batch_files:
            with open(os.path.join(OUTPUT_FOLDER, bf), "r", encoding="utf-8") as infile:
                outfile.write(infile.read() + "\n\n")

    print(f"🎉 All done! Final file: {FINAL_OUTPUT}")

if __name__ == "__main__":
    process_all()
