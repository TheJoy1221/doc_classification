# PDFMaster - Cleaned & GitHub-Ready Version
import os
import csv
import shutil
import logging
from pathlib import Path
from dotenv import load_dotenv
import openai
import pytesseract
import pdfplumber
import PyPDF2
import pypdfium2
import slate3k as slate
import tabula
import pdf2image
import torch
from PIL import Image
from transformers import BertTokenizer, BertForSequenceClassification
from PyPDF2 import PdfReader, PdfWriter

# Load environment variables
load_dotenv()

# Configuration from environment
openai.api_key = os.getenv("OPENAI_API_KEY")
tesseract_path = os.getenv("TESSERACT_PATH")
model_path = os.getenv("MODEL_PATH")
input_folder = os.getenv("INPUT_FOLDER")
output_folder_invoices = os.getenv("OUTPUT_INVOICES")
output_folder_other = os.getenv("OUTPUT_OTHER")
archive_folder = os.getenv("ARCHIVE_FOLDER")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Load model
try:
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    logging.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Model loading failed: {e}")
    raise

# Text extraction fallbacks
def extract_text_from_pdf(pdf_path):
    for extractor in [extract_pdfplumber, extract_pypdf2, extract_pypdfium, extract_slate, extract_tabula, extract_tesseract]:
        text = extractor(pdf_path)
        if text:
            return text
    return None

def extract_pdfplumber(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return ''.join(p.extract_text() or '' for p in pdf.pages)
    except Exception as e:
        logging.warning(f"pdfplumber failed: {e}")
        return None

def extract_pypdf2(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return ''.join(p.extract_text() or '' for p in reader.pages)
    except Exception as e:
        logging.warning(f"PyPDF2 failed: {e}")
        return None

def extract_pypdfium(pdf_path):
    try:
        pdf = pypdfium2.PdfDocument(pdf_path)
        text = ''.join(page.get_textpage().get_text_range() for page in pdf)
        pdf.close()
        return text
    except Exception as e:
        logging.warning(f"pypdfium2 failed: {e}")
        return None

def extract_slate(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            return '\n'.join(slate.PDF(f))
    except Exception as e:
        logging.warning(f"slate failed: {e}")
        return None

def extract_tabula(pdf_path):
    try:
        tables = tabula.read_pdf(pdf_path, pages='all')
        return '\n'.join(str(t) for t in tables)
    except Exception as e:
        logging.warning(f"tabula failed: {e}")
        return None

def extract_tesseract(pdf_path):
    try:
        images = pdf2image.convert_from_path(pdf_path)
        return ''.join(pytesseract.image_to_string(img) for img in images)
    except Exception as e:
        logging.warning(f"Tesseract failed: {e}")
        return None

def classify_document(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()

def gpt_classify(text):
    prompt = f"Classify the following document as either 'Invoice' or 'Other':\n\n{text}\n\nClassification:"
    try:
        response = openai.Completion.create(model="gpt-3.5-turbo", prompt=prompt, max_tokens=50)
        result = response.choices[0].text.strip().lower()
        return 1 if result == 'invoice' else 0
    except Exception as e:
        logging.warning(f"GPT classification failed: {e}")
        return 0

def log_classification(path, initial, refined, correct):
    with open("classification_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([path, initial, refined, correct])

def move_file(src, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(src, dest)

def split_pdfs(folder):
    folder = Path(folder)
    for file in folder.glob("*.pdf"):
        try:
            reader = PdfReader(str(file))
            for i, page in enumerate(reader.pages):
                writer = PdfWriter()
                writer.add_page(page)
                new_name = file.stem + f"_page_{i + 1}.pdf"
                with open(folder / new_name, "wb") as f:
                    writer.write(f)
            logging.info(f"Split {file.name} successfully.")
        except Exception as e:
            logging.error(f"Failed to split {file.name}: {e}")

def process_and_classify(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    if not text:
        logging.warning(f"No text extracted from {pdf_path}")
        return
    initial = classify_document(text)
    refined = gpt_classify(text) if initial == 0 else initial
    target = output_folder_invoices if refined == 1 else output_folder_other
    move_file(pdf_path, os.path.join(target, os.path.basename(pdf_path)))
    log_classification(pdf_path, initial, refined, refined)  # Assumes GPT accuracy or user input system is replaced

def process_all(folder):
    results = []
    for pdf in os.listdir(folder):
        if pdf.endswith(".pdf"):
            path = os.path.join(folder, pdf)
            process_and_classify(path)
    logging.info("Classification complete.")

def archive_remaining():
    for file in Path(input_folder).glob("*.pdf"):
        move_file(file, Path(archive_folder) / file.name)

# Main execution
os.makedirs(output_folder_invoices, exist_ok=True)
os.makedirs(output_folder_other, exist_ok=True)
os.makedirs(archive_folder, exist_ok=True)

logging.info("Starting PDF splitting...")
split_pdfs(input_folder)
logging.info("Starting classification...")
process_all(input_folder)
archive_remaining()