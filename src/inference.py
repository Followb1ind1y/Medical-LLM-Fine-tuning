import re
import torch
from transformers import GenerationConfig
from sklearn.metrics import classification_report
from rouge import Rouge
from bert_score import score as bert_score

def format_prompt(context, question):
    template = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a clinical expert. Your task is to analyze the given medical literature context "
        "and then provide a Final Decision and a Long Answer. "
        f"Context: {context}\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Question: {question}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return template

def refine_output(raw_text):
    # 查找 "user" 和 "end_header_id" 标签之间的内容
    start_index = raw_text.find('<|start_header_id|>assistant<|end_header_id|>')
    
    # 提取并打印相应的内容
    refined_text = raw_text[start_index:].strip()
    return refined_text

def interactive_test(model, tokenizer, context, question, max_length=512, temperature=0.7):
    # 生成prompt
    full_prompt = format_prompt(context, question)
            
    # Tokenize输入
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_k=40,               # 增加多样性
        top_p=0.85,
        repetition_penalty=1.15,
        max_new_tokens=max_length,
        eos_token_id=tokenizer.eos_token_id,  # 直接使用tokenizer属性
        pad_token_id=tokenizer.pad_token_id   # 显式设置
    )
            
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)
            
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(refine_output(raw_output))
    # print(raw_output)
        