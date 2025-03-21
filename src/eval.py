from transformers import TrainerCallback
from sklearn.metrics import accuracy_score, f1_score
from rouge import Rouge
import re

class DualEvaluationCallback(TrainerCallback):
    def __init__(self, tokenizer, test_dataset):
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.response_template = tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>\n",
            add_special_tokens=False
        ) + tokenizer.encode("\n", add_special_tokens=False)

    def _extract_decision(self, text):
        # 提取分类标签
        match = re.search(r'Final Decision:\s*(\w+)', text, re.IGNORECASE)
        return match.group(1).lower() if match else "maybe"

    def _extract_answer(self, text):
        # 提取生成内容
        parts = text.split("Long Answer:")
        return parts[1].strip() if len(parts) > 1 else ""

    def on_evaluate(self, args, state, control, **kwargs):
        # 生成预测
        model = kwargs.pop('model')
        pred_decisions, pred_answers = [], []
        true_decisions, true_answers = [], []

        for example in self.test_dataset:
            # 生成完整回答
            inputs = self.tokenizer(example["text"], return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=400)
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 解析结果
            pred_decisions.append(self._extract_decision(full_text))
            pred_answers.append(self._extract_answer(full_text))
            true_decisions.append(example["final_decision"].lower())
            true_answers.append(example["long_answer"])

        # 分类评估
        acc = accuracy_score(true_decisions, pred_decisions)
        f1 = f1_score(true_decisions, pred_decisions, average="macro")

        # 生成评估
        rouge = Rouge().get_scores(pred_answers, true_answers, avg=True)

        # 记录指标
        print(f"Classification - Acc: {acc:.4f}, F1: {f1:.4f}")
        print(f"Generation - ROUGE-L: {rouge['rouge-l']['f']:.4f}")

def comprehensive_evaluation(model, tokenizer, test_dataset):
    # 生成所有预测
    model.eval()
    predictions = {"decisions": [], "answers": []}
    ground_truth = {"decisions": [], "answers": []}
    
    for example in test_dataset:
        inputs = tokenizer(example["text"], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=400)
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 解析预测
        pred_decision = re.search(r'Final Decision:\s*(\w+)', full_text, re.I).group(1).lower()
        pred_answer = full_text.split("Long Answer:")[1].strip() if "Long Answer:" in full_text else ""
        
        # 保存结果
        predictions["decisions"].append(pred_decision)
        predictions["answers"].append(pred_answer)
        ground_truth["decisions"].append(example["final_decision"].lower())
        ground_truth["answers"].append(example["long_answer"])
    
    # 分类报告
    print("=== Classification Metrics ===")
    print(classification_report(ground_truth["decisions"], predictions["decisions"]))
    
    # 生成报告
    print("\n=== Generation Metrics ===")
    rouge = Rouge().get_scores(predictions["answers"], ground_truth["answers"], avg=True)
    bert_p, bert_r, bert_f = bert_score(
        predictions["answers"], ground_truth["answers"], lang="en"
    )
    print(f"ROUGE-L: {rouge['rouge-l']['f']:.4f}")
    print(f"BERTScore F1: {bert_f.mean().item():.4f}")