import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredAPIFileLoader

# Load API key from .env file
load_dotenv()
api_key = os.getenv("UNSTRUCTURED_API_KEY")

if not api_key:
    raise ValueError("UNSTRUCTURED_API_KEY not found in .env file")

#  input and output directories
RAW_DIR = Path("data/raw_pdfs")
TEXT_DIR = Path("data/texts")
TEXT_DIR.mkdir(exist_ok=True, parents=True)

def process_all_pdfs():
    for file in RAW_DIR.glob("*.pdf"):
        print(f"📄 Processing: {file.name}")

        # Create loader
        loader = UnstructuredAPIFileLoader(
            file_path=str(file),
            api_key=api_key,
            strategy="hi_res",  # Better accuracy on mixed (text+image) PDFs
        )

        # Load documents from API
        docs = loader.load()

        # Combine all text content
        text = "\n\n".join(doc.page_content for doc in docs)

        # Save to .txt
        text_file = TEXT_DIR / (file.stem + ".txt")
        text_file.write_text(text, encoding="utf-8")

        print(f"✅ Saved text to {text_file.name}")

    print("🎉 All PDFs processed. Check data/texts")

if __name__ == "__main__":
    process_all_pdfs()
# code BELOW using local OCR (commented out for reference)






# import fitz
# from pathlib import Path
# import pdfplumber
# import subprocess
# from pdf2image import convert_from_path
# import pytesseract
# from pypdf import PdfReader
# from PIL import Image
# import io

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# RAW_DIR = Path("data/raw_pdfs")
# PROCESSED_DIR = Path("data/processed_pdfs")
# TEXT_DIR = Path("data/texts")

# PROCESSED_DIR.mkdir(exist_ok=True)
# TEXT_DIR.mkdir(exist_ok=True)


# def needs_ocr(pdf_path, min_chars=30):
#     with pdfplumber.open(pdf_path) as pdf:
#         total_pages = len(pdf.pages)
#         image_pages = 0
#         for i, page in enumerate(pdf.pages):
#             text = page.extract_text() or ""
#             if len(text.strip()) < min_chars:
#                 image_pages += 1
#         print(f"{pdf_path.name}: {image_pages}/{total_pages} pages look like images")
#         return image_pages > 0


# def run_ocr(input_path, output_path):
#     print(f"Running OCR on: {input_path.name}")
#     cmd = [
#         "ocrmypdf",
#         "--force-ocr",
#         "--rotate-pages",
#         "-l", "eng",
#         str(input_path),
#         str(output_path)
#     ]
#     subprocess.run(cmd, check=True)


# def extract_text_with_ocr(pdf_path):
#     print(f"Extracting text (with OCR fallback) from: {pdf_path.name}")
#     reader = PdfReader(pdf_path)
#     doc = fitz.open(pdf_path)
#     all_text = ""

#     for i, page in enumerate(reader.pages):
#         # 1. Try normal text
#         text = page.extract_text() or ""

#         # 2. Also OCR the image version of the same page
#         pix = doc[i].get_pixmap(dpi=300)
#         img = Image.open(io.BytesIO(pix.tobytes("png")))
#         ocr_text = pytesseract.image_to_string(img)

#         combined = text.strip() + "\n" + ocr_text.strip()
#         all_text += f"\n--- Page {i+1} ---\n{combined}"

#     return all_text

# def process_all_pdfs():
#     for file in RAW_DIR.glob("*.pdf"):
#         processed_pdf = PROCESSED_DIR / file.name
#         text_file = TEXT_DIR / (file.stem + ".txt")

#         # Step 1: OCR if needed
#         if needs_ocr(file):
#             run_ocr(file, processed_pdf)
#         else:
#             print(f"No OCR needed: {file.name}")
#             processed_pdf.write_bytes(file.read_bytes())

#         # Step 2: Extract text (always uses OCR fallback)
#         extracted_text = extract_text_with_ocr(processed_pdf)
#         text_file.write_text(extracted_text, encoding="utf-8")
#         print(f"✅ Saved text to {text_file.name}")

#     print("🎉 All PDFs processed. Check data/processed_pdfs and data/texts")


# if __name__ == "__main__":
#     process_all_pdfs()
