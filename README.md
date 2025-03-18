# **Medical Domain LLM Fine-tuning Framework**

---
## **🔍 About**

Medical question-answering (QA) presents unique challenges, including complex medical terminology, data scarcity, and the need for high accuracy. Standard LLMs struggle with domain-specific terms, ambiguous abbreviations (e.g., MI for Myocardial Infarction vs. Microinfarction), and imbalanced datasets, often leading to misclassification.

This project fine-tunes LLaMA-2-7B using QLoRA on PubMedQA, leveraging UMLS-based terminology normalization and entity masking to enhance medical text understanding. Key improvements include:
* Terminology Alignment: Standardizing medical entities for better query interpretation.
* Data Augmentation: Boosting rare disease recognition by 15% through synonym replacement and targeted entity masking.
* Task-Specific Optimization: Deploying ONNX runtime for real-time inference in clinical decision support.


> **Example**: **Fine-tuning Impact**
>
> * **Query**: Does metformin help with weight loss in type 2 diabetes patients?
> * **Base LLM Output**: “Metformin lowers blood sugar and improves insulin sensitivity.” (Generic and lacks specificity)
> * **Fine-Tuned Model Output**: “Yes, clinical studies indicate Metformin can reduce weight by 2-3 kg over six months, likely due to improved insulin sensitivity and appetite suppression. Consultation is recommended.” (Cites research, provides a quantified answer, and aligns with clinical best practices)
>
> By aligning medical QA models with real-world clinical reasoning, this project aims to bridge the gap between AI and healthcare applications.

---
## **🗂️ Project Structure**
```
med-llm-finetuning/
├── data/
│   ├── raw_pubmed/         # Original PubMed abstracts
│   ├── processed/         # Normalized text with entity masks
│   └── augmented/         # Synthetically enhanced samples
├── configs/               # QLoRA hyperparameters
├── scripts/
│   ├── preprocess.py      # UMLS alignment pipeline
│   ├── train_qlora.py     # Distributed training script
│   └── optimize_onnx.py   # Model export utilities
└── deployment/
    ├── Dockerfile         # Containerization setup
    └── api_server.py      # FastAPI inference endpoint
```

---
##