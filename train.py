import torch
import argparse
from configs.lora_config import get_peft_config
from configs.training_args import get_training_args
from src.data_utils import load_and_process_data
from src.modeling import load_model_and_tokenizer
from src.inference import interactive_test
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

def main(args):
    if args.resume_from_checkpoint:
        model, tokenizer = load_model_and_tokenizer(args.resume_from_checkpoint)
    else:
        model, tokenizer = load_model_and_tokenizer()

    if args.test:
        interactive_test(
            model, 
            tokenizer,
            input= args.input,
            max_length=args.max_length,
            temperature=args.temperature
        )
        return


    dataset = load_and_process_data()
    peft_config = get_peft_config()
    training_args = get_training_args(args.output_dir, args.epochs)

    # Data collator
    collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>\n",
            add_special_tokens=False
        ),
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        data_collator=collator,
        args=training_args,
    )

    if args.eval_only:
        trainer.evaluate()
    elif args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 正常训练
    # python train.py --epochs 5 --output_dir ./medqa-model
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./results")
    # 从检查点恢复训练
    # python train.py --output_dir ./medqa-model --resume_from_checkpoint ./medqa-model/checkpoint-1000
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to specific checkpoint directory")
    # 仅评估测试集
    # python train.py --eval_only --resume_from_checkpoint ./medqa-model --epochs 5 --output_dir ./medqa-model
    parser.add_argument("--eval_only", action="store_true", 
                       help="Only run evaluation on the test set")
    # 交互测试模式
    # python train.py --test \--resume_from_checkpoint ./medqa-model \ --max_length 256 \ --temperature 0.7
    parser.add_argument("--test", action="store_true",
                      help="进入交互测试模式")
    parser.add_argument("--input", type=str, default=None,
                      help="问题")
    parser.add_argument("--max_length", type=int, default=512,
                      help="生成文本的最大长度")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="生成温度（0.1-1.0）")
    
    args = parser.parse_args()
    
    try:
        main(args)
    finally:
        # 分布式清理
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        # CUDA缓存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("资源清理完成")