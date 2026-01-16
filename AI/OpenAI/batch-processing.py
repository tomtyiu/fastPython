#Batch processing
#!pip install -q --upgrade torch

#!pip install -q transformers triton==3.5.1 kernels

#!pip uninstall -q torchvision torchaudio -y

model_id = "EpistemeAI/RSI-AI-V1.1"

import torch
from collections import deque
import asyncio
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

class BatchProcessor:
    def __init__(self, model, tokenizer, max_batch_size=32, max_wait_time=0.01):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.request_queue = deque()
        self.processing = False

    async def predict(self, text: str) -> torch.Tensor:
        """Add prediction request to batch queue"""
        future = asyncio.Future()
        self.request_queue.append((text, future))

        if not self.processing:
            asyncio.create_task(self._process_batch())

        return await future

    async def _process_batch(self):
        """Process accumulated requests in batches"""
        self.processing = True

        while self.request_queue:
            # Collect batch
            batch_items = []
            for _ in range(min(self.max_batch_size, len(self.request_queue))):
                if self.request_queue:
                    batch_items.append(self.request_queue.popleft())

            if not batch_items:
                break

            # Extract texts and futures
            texts, futures = zip(*batch_items)

            # Tokenize batch
            inputs = self.tokenizer(
                list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            # Move input tensors to the model's device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Run batch inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Distribute results
            for i, future in enumerate(futures):
                # Set the full outputs object as the result
                future.set_result(outputs)

            # Small delay to allow queue accumulation
            await asyncio.sleep(self.max_wait_time)

        self.processing = False

# Usage example
async def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="cuda",
    )
    processor = BatchProcessor(model, tokenizer)

    # Simulate concurrent requests
    texts = [f"Sample text {i}" for i in range(50)]

    start = time.time()
    tasks = [processor.predict(text) for text in texts]
    results = await asyncio.gather(*tasks)
    batch_time = time.time() - start

    # Compare with individual processing
    start = time.time()
    individual_results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)
        individual_results.append(output)
    individual_time = time.time() - start

    print(f"Individual processing: {individual_time:.2f}s")
    print(f"Batch processing: {batch_time:.2f}s")
    print(f"Throughput improvement: {individual_time/batch_time:.2f}x")

# Run the async example
# asyncio.run(main())

!pip install -q nest_asyncio
import nest_asyncio
nest_asyncio.apply()

import time
import asyncio # Import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM

sample_text = "What is pi?."

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cuda"
)

# Initialize the BatchProcessorsi
processor = BatchProcessor(model, tokenizer)

# Define an async function to perform inference
async def perform_inference(text):
    outputs = await processor.predict(text)
    return outputs

# Run the inference using asyncio.run() after applying nest_asyncio
output_result = asyncio.run(perform_inference(sample_text))

print("Inference output type:", type(output_result))
print("Inference output logits shape:", output_result.logits.shape)


messages = [
    {"role": "system", "content": "You are helpful math assistant"},
    {"role": "user", "content": "What is C. elegans?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="high",
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=2048)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
