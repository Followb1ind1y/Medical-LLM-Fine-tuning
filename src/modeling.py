import os
import torch
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# 从环境变量获取token（推荐在.bashrc中设置export HF_TOKEN=your_token）
hf_token = os.getenv("HF_TOKEN")
assert hf_token, "必须设置HF_TOKEN环境变量"
# 登录并缓存凭证
login(token=hf_token)

def load_model_and_tokenizer(model_name="meta-llama/Meta-Llama-3-8B"):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|begin_of_text|>", "<|eot_id|>"]
    })
    tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    return model, tokenizer