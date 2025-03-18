from datasets import load_dataset

def load_and_process_data(dataset_name="qiaojin/PubMedQA", test_size=0.1):
    dataset = load_dataset(dataset_name, "pqa_labeled")
    dataset = dataset["train"].train_test_split(test_size=test_size, seed=42)
    
    def format_instruction(example):
        # Custom system prompt for medical domain
        system_msg = "You are an AI assistant with a PhD in clinical medicine, required to answer questions based on evidence-based medicine."

        # Structured user instruction
        user_input = (
            f"Question: {example['question']}\n"
            "Please analyze key evidence from the literature first, then provide step-by-step explanation, finally give diagnosis starting with 'Conclusion:'"
        )

        # Structured model response
        assistant_response = (
            f"Final Decision: {example['final_decision']}\n"
            f"Long Answer:{example['long_answer']}"
        )

        # Using official LLaMA3 template
        return {
            "text": (
                f"<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n{system_msg}<|eot_id|>\n"
                f"<|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>\n"
                f"<|start_header_id|>assistant<|end_header_id|>\n{assistant_response}<|eot_id|>\n"
            )
        }
    
    return dataset.map(
        format_instruction,
        remove_columns=["pubid", "context", "question", "long_answer", "final_decision"]
    )