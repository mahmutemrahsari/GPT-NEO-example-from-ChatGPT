from transformers import pipeline, set_seed

# Load the GPT-Neo model
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')

# Set the seed for reproducibility
set_seed(84)

# Generate text
generated_text = generator('What is life ?', max_length=1250, do_sample=True)[0]['generated_text']

# Print the generated text
print(generated_text)
