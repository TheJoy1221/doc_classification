from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os

# Enhanced dataset - ensure a diverse and larger sample
data = {
    "text": [
        "1 HEALTH INSURANCE CLAIM FORM",
        "1 Physician Supplier Information",
        "Care Report with patient details and medical history",
        "Fax Cover Sheet with contact information and date",
        "1 PATIENT AND INSURED INFORMATION",
        "Final report with medical conclusions",
        "1 UB-04 form for hospital billing",
        "1 Itemized Charges for medical services",
        "1 NUCC form for healthcare providers",
        "1 Invoice for medical services rendered",
        "Physical Therapy report for patient progress",
        "Assessment report of patient's condition",
        "1 Approved OMB",
        "1 Remit Payment to healthcare provider",
        "Doctor's note for medical recommendations",
        "1 Receipt for payment received",
        "1 Total Charges for the medical procedures",
        "1 Invoice Date for the billing period",
        "History of patient's medical visits",
        "1 NUCC Instruction Manual available",
        "1 Current Balance",
        "1 Hospital Charges",
        "1 REBILL",
        "Notes:",
        "Vitals",
        "Medical History",
        "Transcription",
        "Discussion/Summary",
        "Radiology Report",
        "1 CPT/HCPCS",
        "1 Diagnosis Pointer",
        "Reason for Visit",
        "Surgical History",
        "THERAPY PARTNERS",
        "1 SpeakEasy Translation & Transportation",
        "Patient Care Record",
        "1 BILLING PROVIDER INFO",
        "Work Restrictions",
        "Patient Clinical Summary",
        "1 Procedure(s) and Charge(s)",
        "1 Amount Due",
        "1 Statement of Pharmacy Services",
        "1 Invoice Number",
        "1 Itemized Statement",
        "Texas Workers' Compensation Work Status Report",
        "Report of Medical Evaluation",
        "Physical Therapy Treatment Note",
        "Physical Therapy Daily Note",
        "Activity Comments",
        "Visit History",
        "ACTIVITY LOG",
        "Topics Discussed",
        "Subjective Examination",
        "Physician's Inquiry",
        "Transportation Summary",
        "1 APPEAL",
        "WORK STATUS INFORMATION",
        "Medical Status Information",
        "ED Orders",
        "1 TOTAL CHARGE",
        "Exercise Activities",
        "1 SUMMARY OF CHARGES",
        "1 AMOUNT PAID",
        # Add more diverse examples here
    ],
    "labels": [
        1,  # "1 HEALTH INSURANCE CLAIM FORM"
        1,  # "1 Physician Supplier Information"
        0,  # "Care Report with patient details and medical history"
        0,  # "Fax Cover Sheet with contact information and date"
        1,  # "1 PATIENT AND INSURED INFORMATION"
        0,  # "Final report with medical conclusions"
        1,  # "1 UB-04 form for hospital billing"
        1,  # "1 Itemized Charges for medical services"
        1,  # "1 NUCC form for healthcare providers"
        1,  # "1 Invoice for medical services rendered"
        0,  # "Physical Therapy report for patient progress"
        0,  # "Assessment report of patient's condition"
        1,  # "1 Approved OMB"
        1,  # "1 Remit Payment to healthcare provider"
        0,  # "Doctor's note for medical recommendations"
        1,  # "1 Receipt for payment received"
        1,  # "1 Total Charges for the medical procedures"
        1,  # "1 Invoice Date for the billing period"
        0,  # "History of patient's medical visits"
        1,  # "1 NUCC Instruction Manual available"
        1,  # "1 Current Balance"
        1,  # "1 Hospital Charges"
        1,  # "1 REBILL"
        0,  # "Notes:"
        0,  # "Vitals"
        0,  # "Medical History"
        0,  # "Transcription"
        0,  # "Discussion/Summary"
        0,  # "Radiology Report"
        1,  # "1 CPT/HCPS"
        1,  # "1 Diagnosis Pointer"
        0,  # "Reason for Visit"
        0,  # "Surgical History"
        0,  # "THERAPY PARTNERS"
        1,  # "1 SpeakEasy Translation & Transportation"
        0,  # "Patient Care Record"
        1,  # "1 BILLING PROVIDER INFO"
        0,  # "Work Restrictions"
        0,  # "Patient Clinical Summary"
        1,  # "1 Procedure(s) and Charge(s)"
        1,  # "1 Amount Due"
        1,  # "1 Statement of Pharmacy Services"
        1,  # "1 Invoice Number"
        1,  # "1 Itemized Statement"
        0,  # "Texas Workers' Compensation Work Status Report"
        0,  # "Report of Medical Evaluation"
        0,  # "Physical Therapy Treatment Note"
        0,  # "Physical Therapy Daily Note"
        0,  # "Activity Comments"
        0,  # "Visit History"
        0,  # "ACTIVITY LOG"
        0,  # "Topics Discussed"
        0,  # "Subjective Examination"
        0,  # "Physician's Inquiry"
        0,  # "Transportation Summary"
        1,  # "1 APPEAL"
        0,  # "WORK STATUS INFORMATION"
        0,  # "Medical Status Information"
        0,  # "ED Orders"
        1,  # "1 TOTAL CHARGE"
        0,  # "Exercise Activities"
        1,  # "SUMMARY OF CHARGES"
        1,  # "AMOUNT PAID"
        # Match labels to the additional examples
    ]
}

# Ensure the length of labels matches the length of text
assert len(data["text"]) == len(data["labels"]), "Number of texts and labels must match"

dataset = Dataset.from_dict(data)

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


# Tokenize dataset
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)


tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Split dataset into training and validation
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train model
trainer.train()

# Save the trained model and tokenizer
model_save_path = r"C:\Users\adavila\Documents\myenv\document_classifier_model"
os.makedirs(model_save_path, exist_ok=True)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Verify the contents of the model directory
saved_files = os.listdir(model_save_path)
print("Contents of model directory:", saved_files)

# Check if files are actually created
for file_name in saved_files:
    file_path = os.path.join(model_save_path, file_name)
    print(f"{file_name} exists: {os.path.exists(file_path)}")
