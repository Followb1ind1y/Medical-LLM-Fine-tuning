import re
import torch
from transformers import GenerationConfig
from sklearn.metrics import classification_report
from rouge import Rouge
from bert_score import score as bert_score

def format_prompt(user_input: str):
    template = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are an AI assistant with a PhD in clinical medicine, required to answer questions based on evidence-based medicine.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Question: {user_input}\n"
        "Please analyze key evidence from the literature first, then provide step-by-step explanation, finally give diagnosis starting with 'Conclusion:'<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return template

def refine_output(raw_text):
    # 提取问题部分
    question_match = re.search(
        r'<\|start_header_id\|>user<\|end_header_id\|>\nQuestion: (.*?)\nPlease analyze', 
        raw_text, 
        re.DOTALL
    )
    question = question_match.group(1).strip() if question_match else "N/A"
    
    # 提取assistant回答
    answer_match = re.search(
        r'<\|start_header_id\|>assistant<\|end_header_id\|>\n(.*?)(<\|eot_id\||<\|end_of_text\||$)', 
        raw_text, 
        re.DOTALL
    )
    answer = answer_match.group(1).strip() if answer_match else ""
    
    # 格式标准化
    formatted = f"Question: {question}\n\n"  # 添加空行分隔
    
    # 提取Final Decision
    decision_match = re.search(r'Final Decision:?\s*([^\n]+)', answer)
    decision = decision_match.group(1).strip() if decision_match else "Undetermined"
    
    # 提取Long Answer
    long_answer = re.sub(r'Final Decision:.*?\n', '', answer, flags=re.DOTALL)
    long_answer = re.sub(r'\n+', ' ', long_answer).strip()  # 合并多余换行
    
    # 标准化首字母大写
    decision = decision[0].upper() + decision[1:].lower()
    
    # 生成最终格式
    return (
        f"Question: {question}\n\n"
        f"Final Decision: {decision}\n"
        f"Long Answer: {long_answer}"
    )

def interactive_test(model, tokenizer, input, max_length=512, temperature=0.7):

    user_input = input
                
    # 生成prompt
    full_prompt = format_prompt(user_input)
            
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
    # print(refine_output(raw_output))
    print(raw_output)
        