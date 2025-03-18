from transformers import TrainingArguments

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./med-qa-llama"

# Number of training epochs
num_train_epochs = 10

# Batch size per GPU for training
per_device_train_batch_size = 2

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 4

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Optimizer to use
optim = "paged_adamw_32bit"

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Learning rate schedule
lr_scheduler_type = "cosine"

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 100

# Log every X updates steps
logging_steps = 50

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = True
bf16 = False

# evaluation_strategy = "steps"
# eval_steps = 20

# Set training parameters
def get_training_args(output_dir="./output", num_train_epochs=1):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        # per_device_train_batch_size=per_device_train_batch_size,
        # gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        optim=optim,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=logging_steps,
        # evaluation_strategy=evaluation_strategy,
        # eval_steps=eval_steps,
        report_to="tensorboard",
        fp16=fp16,
        bf16=bf16,
        group_by_length=group_by_length,
        deepspeed="./configs/deepspeed_z3.json"
    )