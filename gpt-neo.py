from transformers import pipeline, set_seed
import torch
import time

start_time = time.time()

# Check if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")



# Load the GPT-Neo model
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device=device)

# Set the seed for reproducibility
set_seed(42)

# Generate text
generated_text = generator('Who am I?', max_length=400, do_sample=True)[0]['generated_text']

# Get the amount of GPU memory currently in use
memory_used = torch.cuda.memory_allocated(device) / 1024 ** 2
print(f"GPU memory used: {memory_used:.2f} MB")

# Get the maximum amount of GPU memory that has been reserved by PyTorch
memory_reserved = torch.cuda.memory_reserved(device) / 1024 ** 2
print(f"GPU memory reserved: {memory_reserved:.2f} MB")

# Print the generated text
print(generated_text)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
