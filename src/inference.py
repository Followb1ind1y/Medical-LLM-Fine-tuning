import torch

def format_prompt(user_input: str, system_message: str = "You are an AI assistant with a PhD in clinical medicine, required to answer questions based on evidence-based medicine.") -> str:
    """
    将用户输入转换为 LLaMA 官方的 prompt 格式。
    
    :param user_input: 用户输入的问题或请求
    :param system_message: 系统指令，默认为通用 AI 助手
    :return: 格式化的 prompt 字符串
    """
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_message}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n{user_input}\nPlease analyze key evidence from the literature first, then provide step-by-step explanation, finally give diagnosis starting with 'Conclusion:'<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"""
    return prompt

def interactive_test(model, tokenizer, input, max_length=512, temperature=0.7):
    """交互式测试模式
    
    Args:
        model: 加载的模型
        tokenizer: 分词器
        max_length: 生成最大长度
        temperature: 生成温度系数
    """

    user_input = input
                
    # 生成prompt
    full_prompt = format_prompt(user_input)
            
    # Tokenize输入
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
            
    # 提取并打印回复
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n模型回复: {response}")
        