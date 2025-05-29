# 🧠 PDFMaster

**PDFMaster** is an intelligent document processing pipeline for scanned and text-based PDFs. It uses a hybrid transformer + GPT-based approach to classify documents as **Invoices** or **Other**, with fallback OCR strategies and audit logging.

---

## 🚀 Features

- 🔍 Multi-fallback PDF parsing: `pdfplumber`, `PyPDF2`, `pypdfium2`, `OCR`, etc.
- 🤖 BERT-based classification (custom-trained)
- 🧠 GPT secondary classification refinement
- 📦 File sorting & movement into `/Invoices` and `/Other`
- 📜 Logs classifications in CSV for transparency
- 🔐 Configurable via `.env`

---

## 📂 Project Structure

```
pdfmaster/
├── pdfmaster.py                  # Main automation script
├── train_invoice_classifier.py  # BERT training script
├── .env.example                 # Environment config template
├── README.md
├── requirements.txt
└── /models                      # Folder for storing trained model (not included)
```

---

## 🔧 Setup

1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/pdfmaster.git
   cd pdfmaster
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Create `.env`  
   ```bash
   cp .env.example .env
   ```

4. Run training (optional, model included separately)  
   ```bash
   python train_invoice_classifier.py
   ```

5. Run main script  
   ```bash
   python pdfmaster.py
   ```

---

## 🛠️ Environment Configuration

| Variable            | Description                           |
|---------------------|---------------------------------------|
| OPENAI_API_KEY      | Your OpenAI API key                   |
| TESSERACT_PATH      | Full path to your `tesseract.exe`     |
| MODEL_PATH          | Folder path to load/save model        |
| INPUT_FOLDER        | Directory with raw PDFs               |
| OUTPUT_INVOICES     | Destination folder for invoices       |
| OUTPUT_OTHER        | Destination folder for other files    |
| ARCHIVE_FOLDER      | Folder to archive processed files     |

---

## 📘 Model Training Notes

This repo includes `train_invoice_classifier.py` to generate your own BERT model.
Ensure `transformers` and `datasets` are installed, and customize the dataset in-script.

---

## ✨ Built With Love

Crafted by Amanda with precision, purpose, and a spark of magic.
