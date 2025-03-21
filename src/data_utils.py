from datasets import load_dataset

def load_and_process_data(dataset_name="qiaojin/PubMedQA", test_size=0.1):
    dataset = load_dataset(dataset_name, "pqa_labeled")
    dataset = dataset["train"].train_test_split(test_size=test_size, seed=42)
    
    def format_instruction(example):
        template = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n{system_msg}\n{context_msg}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n{assistant_response}<|eot_id|>"
        )
        return {
            "text": template.format(
                system_msg=(
                    "You are a clinical expert. Your task is to analyze the given medical literature context "
                    "and then provide a Final Decision and a Long Answer. "
                ),
                context_msg = f"Context: {' '.join(example['context']['contexts'])}\n",
                user_input=f"Question: {example['question']}\n",
                assistant_response=f"Final Decision: {example['final_decision']}\nLong Answer:{example['long_answer']}"
            )
        }
    
    return dataset.map(
        format_instruction,
    )