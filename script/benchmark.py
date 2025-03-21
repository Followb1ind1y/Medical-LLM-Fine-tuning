import time
import requests
import numpy as np

def test_vllm():
    url = "http://localhost:5000/generate"
    payload = {
        "prompt": "Percutaneous ethanol injection for benign cystic thyroid nodules: is aspiration of ethanol-mixed fluid advantageous?",
        "max_tokens": 256,
        "temperature": 0.3
    }
    
    # 测试100次请求
    latencies = []
    for _ in range(100):
        start = time.time()
        res = requests.post(url, json=payload)
        latencies.append(time.time() - start)
    
    print(f"[vLLM] Throughput: {100/np.sum(latencies):.1f} req/s")
    print(f"[vLLM] P95 Latency: {np.percentile(latencies, 95):.2f}s")

def test_huggingface():
    from transformers import pipeline
    generator = pipeline("text-generation", model="./hf_model")
    
    start = time.time()
    for _ in range(10):  # 原生实现较慢，减少测试次数
        generator("Percutaneous ethanol injection for benign cystic thyroid nodules: is aspiration of ethanol-mixed fluid advantageous?", max_length=256)
    
    avg_time = (time.time() - start)/10
    print(f"[HuggingFace] Throughput: {1/avg_time:.1f} req/s")
    print(f"[HuggingFace] Avg Latency: {avg_time:.2f}s")

if __name__ == "__main__":
    test_vllm()
    test_huggingface()