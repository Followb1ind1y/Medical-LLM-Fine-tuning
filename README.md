# **Medical Domain LLM Fine-tuning Framework**

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

## **🗂️ Project Structure**
```
med-llm-finetuning/
├── congifs/
│   ├── lora_config.py
│   ├── training_config.py
│   └── deepspeed_z3.json
├── src/
│   ├── data_utils.py
│   ├── inference.py
│   └── modeling.py
├── PubMedQA_Fine_Tuning.ipynb
├── environment.yml
├── train.py
```

## **📦 Enviroment Setup**
```
conda env create -f environment.yml
conda activate med-llm
```

## **🚀 Workflow**

### **1. Colab Prototyping**  <a href="https://colab.research.google.com/drive/1nTfURgLHIdXFTVDZsoKdOugEvxmHBAkB?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"/></a>
**Objective**: Rapidly validate the QLoRA fine-tuning pipeline on a PubMedQA subset using free Colab GPUs (T4). 

* **Dataset Curation**: Processed PubMedQA into Llama-3 instruction format while preserving clinical context; applied stratified 90/10 train-test splits to maintain label distribution.
* **QLoRA Configuration**: Initialized 4-bit quantization with `BitsAndBytesConfig` using `compute_dtype=bfloat16` for stability; optimized LoRA rank (`r=16`) and scaling (`alpha=32`) via ablation studies on the data.
* **Training Pipeline**: Engineered `SFTTrainer` with gradient checkpointing and small batch size to fit T4 VRAM constraints; validated convergence through loss reduction.
* **Checkpoint Reliability**: Ensured fault tolerance by testing resume-from-checkpoint functionality (`trainer.train(resume_from_checkpoint=True)`).
* **Production Readiness**: Modularized code into configurable components (`modeling.py`, `data_utils.py`) for seamless cloud migration; verified CLI execution (`!python train.py --epochs 1 --eval_only`) before deployment.

### **2. Distributed Training (Lambda Lab + DeepSpeed)**
Scale training to multiple GPUs using DeepSpeed ZeRO-3 for memory optimization.

* DeepSpeed Integration:
    ```
    // configs/deepspeed_z3.json
    {
    "fp16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"}
    }
    }
    ```
* Launch Command:
    ```
    deepspeed --num_gpus=8 train.py \
    --deepspeed configs/deepspeed_z3.json \
    --model_name meta-llama/Llama-2-7b-hf \
    --use_peft \
    --peft_method lora
    ```

### **3. Model Evaluation & Optimization**
Evaluate using PubMedQA’s official metrics (Accuracy@3, F1) and optimize for clinical relevance.

### **4. Deployment**
Deploy the quantized model for low-latency inference in clinical environments.

## **sudo Notes**

```
nvcc --version
nvidia-smi

# 下载ARM64版本
<!-- wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh -->

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

conda create -n myenv python=3.11 -y
conda init
conda activate myenv

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

pip install transformers bitsandbytes peft accelerate deepspeed trl datasets huggingface-hub tqdm tensorboard

export HF_TOKEN="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"

conda env export > environment.yml
conda env create -f environment.yml

watch -n 1 nvidia-smi

cd /home/ubuntu/Medical-LLM-Fine-tuning/medqa-model
tensorboard --logdir runs/ --port 6006 --host 0.0.0.0 --reload_interval 10
```

```
deepspeed --num_gpus 2 --master_port 29500 train.py --epochs 5 --output_dir ./medqa-model
```