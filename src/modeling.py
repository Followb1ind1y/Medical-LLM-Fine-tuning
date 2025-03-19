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
    # Add special tokens
    new_tokens = ["<|eot_id|>"]  # 只添加真正的新token
    tokenizer.add_special_tokens({
        "additional_special_tokens": new_tokens
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
        # device_map="auto",
        trust_remote_code=True
    )
    # 分布式训练需要的手动设备分配
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = model.to(f"cuda:{local_rank}")
    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer