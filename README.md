# **Medical Domain LLM Fine-tuning Framework**

## **🔍 About**

Fine-tuning large language models (LLMs) for medical question-answering (QA) presents unique challenges, such as domain-specific terminology, data scarcity, and the necessity for high clinical accuracy. Standard LLMs often misinterpret ambiguous abbreviations (e.g., “RA” for Rheumatoid Arthritis vs. Right Atrium) or struggle with long-form reasoning required in clinical settings.

This project fine-tunes LLaMA-3-8B on PubMedQA using QLoRA for efficient adaptation, optimizing both classification and long-answer generation. The workflow integrates scalable training, robust evaluation, and optimized deployment for real-world clinical inference. Key improvements include:
* **QLoRA Fine-Tuning on PubMedQA**: Efficient low-rank adaptation with 4-bit quantization, optimized for clinical decision-making.
* **Distributed Training with DeepSpeed**: Multi-GPU scaling on Lambda Cloud (2×A100 40GB) with memory-efficient CPU offloading.
* **Task-Specific Evaluation**: Beyond accuracy/F1, measured long-answer coherence using ROUGE and BERTScore to optimize clinical relevance.
* **vLLM-Accelerated Inference**: Deployed with PagedAttention for low-latency, high-throughput medical text generation.


<!-- > **Example**: **Fine-tuning Impact**
>
> * **Query**: Does metformin help with weight loss in type 2 diabetes patients?
> * **Base LLM Output**: “Metformin lowers blood sugar and improves insulin sensitivity.” (Generic and lacks specificity)
> * **Fine-Tuned Model Output**: “Yes, clinical studies indicate Metformin can reduce weight by 2-3 kg over six months, likely due to improved insulin sensitivity and appetite suppression. Consultation is recommended.” (Cites research, provides a quantified answer, and aligns with clinical best practices)
>
> By aligning medical QA models with real-world clinical reasoning, this project aims to bridge the gap between AI and healthcare applications. -->

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
**Objective**: Scale training to multiple GPUs using DeepSpeed ZeRO-3 for memory optimization. A 2×A100 (40GB) multi-GPU instance was deployed using Lambda Cloud, and DeepSpeed ZeRO-3 with CPU offloading was implemented to reduce the per-GPU memory footprint. 

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
    deepspeed --num_gpus=2 train.py \
    --master_port 29500 train.py --epochs 50 \
    --output_dir ./medqa-model 
    ```

### **3. Model Evaluation & Optimization**
**Objective**: Evaluate using PubMedQA’s official metrics (Accuracy, F1) while also assessing long-answer generation quality with ROUGE and BERTScore to optimize for clinical relevance.

* **Fine-Tuned Model Output Example**:
    ```
    Question: Percutaneous ethanol injection for benign cystic thyroid nodules: is aspiration of ethanol-mixed fluid advantageous?
    Context: We evaluated the differences between percutaneous ethanol injection with and without aspiration of ethanol-mixed fluid for treatment ...
    
    Final Decision: No
    Long Answer: Percutaneous ethanol injection (PEI) is an effective treatment for benign cystic thyroid nodules, but the advantage of aspirating ethanol-mixed fluid remains unclear. Some studies suggest that aspiration may reduce ethanol diffusion and improve therapeutic outcomes, while others find no significant difference in efficacy. Further research is needed to determine its clinical benefit.
    ```

* **Classification** 
    | Metric                | Base LLAMA-3-8B | Fine-Tuned Model |
    |-----------------------|----------------|------------------|
    | Accuracy(%)         | 62.8           | 78.1            |
    | Macro-F1(%)           | 58.4            | 73.6             |

* **Generation**
    | Metric                | Base LLAMA-3-8B | Fine-Tuned Model |
    |-----------------------|----------------|------------------|     
    | ROUGE (F1)       | 0.412           | 0.587            |
    | BERTScore (F1)     | 0.661           | 0.723            |

### **4. Deployment**
**Objective**: Make the quantized model ready for low-latency inference in clinical environments.
* **Accelerated Inference with vLLM**: Implemented vLLM to enable high-throughput, low-latency inference, leveraging PagedAttention for optimized memory management.
* **Containerized Deployment**: Encapsulated the inference pipeline into a Docker container, ensuring a portable and scalable solution across cloud platforms.


## **📃 Licence**

This repository is licensed under the Apache-2.0 License - see the [LICENSE](https://github.com/Followb1ind1y/Medical-LLM-Fine-tuning/LICENSE) file for details.