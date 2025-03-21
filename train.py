import torch
import argparse
from configs.lora_config import get_peft_config
from configs.training_args import get_training_args
from src.data_utils import load_and_process_data
from src.modeling import load_model_and_tokenizer
from src.inference import interactive_test
from src.eval import DualEvaluationCallback, comprehensive_evaluation
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from vllm import LLM

# import os
# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
# os.environ["RANK"] = "0"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"

def main(args):
    if args.resume_from_checkpoint:
        model, tokenizer = load_model_and_tokenizer(args.resume_from_checkpoint)
    else:
        model, tokenizer = load_model_and_tokenizer()

    dataset = load_and_process_data()

    if args.test:
        my_context = ' '.join(dataset["test"][args.input]['context']['contexts'])
        my_question = dataset["test"][args.input]['question']
        interactive_test(
            model, 
            tokenizer,
            context= my_context,
            question=my_question,
            max_length=args.max_length,
            temperature=args.temperature
        )
        return

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
    trainer.add_callback(DualEvaluationCallback(tokenizer, dataset["test"]))

    if args.eval_only:
        trainer.evaluate()
        comprehensive_evaluation(trainer.model, tokenizer, dataset["test"])
    elif args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if not args.eval_only and not args.test:
        # 合并LoRA权重
        model = model.merge_and_unload()
        
        # 保存HuggingFace格式
        model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        
        vllm_model = LLM(model=args.output_dir, 
                        tokenizer=args.output_dir,
                        quantization="awq",
                        enforce_eager=True)  # 兼容性模式
        vllm_model.save("./vllm_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 正常训练
    # python train.py --epochs 5 --output_dir ./medqa-model
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./results")
    # 从检查点恢复训练
    # python train.py --output_dir ./medqa-model --resume_from_checkpoint ./medqa-model/checkpoint-224
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to specific checkpoint directory")
    # 仅评估测试集
    # python train.py --eval_only --resume_from_checkpoint ./medqa-model --epochs 5 --output_dir ./medqa-model
    parser.add_argument("--eval_only", action="store_true", 
                       help="Only run evaluation on the test set")
    # 交互测试模式
    # python train.py --test --input 1 --max_length 256 --temperature 0.7
    parser.add_argument("--test", action="store_true",
                      help="进入交互测试模式")
    parser.add_argument("--input", type=int, default=0,
                      help="问题")
    parser.add_argument("--max_length", type=int, default=512,
                      help="生成文本的最大长度")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="生成温度（0.1-1.0）")
    
    # DeepSpeed
    # deepspeed --num_gpus 2 --master_port 29500 train.py --epochs 5 --output_dir ./medqa-model
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank passed from distributed launcher")
    
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