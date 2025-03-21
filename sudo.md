## **sudo Notes**

```
nvcc --version
nvidia-smi

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

conda create -n myenv python=3.11 -y
conda init
conda activate myenv

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

pip install transformers bitsandbytes peft accelerate deepspeed trl datasets huggingface-hub tqdm tensorboard rouge-score bert-score scikit-learn rouge

export HF_TOKEN="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"

conda env export > environment.yml
conda env create -f environment.yml

watch -n 1 nvidia-smi

cd /home/ubuntu/Medical-LLM-Fine-tuning/medqa-model
tensorboard --logdir runs/ --port 6006 --host 0.0.0.0 --reload_interval 10
```

```
deepspeed --num_gpus 2 --master_port 29500 train.py --epochs 5 --output_dir ./medqa-model
```