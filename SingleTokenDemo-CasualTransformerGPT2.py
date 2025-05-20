from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Step 1: Load the GPT-2 tokenizer and model from Hugging Face
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()  # Set model to inference mode (disables dropout, etc.)

# Step 2: Define your input prompt
text = "I love you. Winston"
inputs = tokenizer(text, return_tensors="pt")  # Tokenize input and convert to PyTorch tensors

# Print tokenization info
input_ids = inputs["input_ids"]  # Shape: [1, seq_len]
print("=== Tokenization ===")
print("Input text:", text)
print("Input IDs:", input_ids)  # Tensor of token indices
print("Tokens:", tokenizer.convert_ids_to_tokens(input_ids[0]))  # Human-readable token list
print()

# Step 3: Run the model forward pass to get output logits
# Logits represent unnormalized scores for each possible next token at each input position
with torch.no_grad():  # Disable gradient computation for faster inference
    outputs = model(**inputs)
    logits = outputs.logits  # Shape: [1, seq_len, vocab_size]

# Step 4: Focus on the logits of the **last input token**
# These logits predict what token is likely to come next
last_token_logits = logits[0, -1, :]  # Shape: [vocab_size]

# Apply temperature to control the randomness of predictions
# Lower = more deterministic, Higher = more random
temperature = 1.0
adjusted_logits = last_token_logits / temperature

# Step 5: Convert logits into a probability distribution using softmax
probs = torch.softmax(adjusted_logits, dim=0)  # Shape: [vocab_size]

# Step 6: Find the top-k most probable next tokens
top_k = 10
top_probs, top_indices = torch.topk(probs, top_k)

# Print the top-k predictions and their probabilities
print(f"=== Top {top_k} Tokens (temperature={temperature}) ===")
for i in range(top_k):
    token_id = top_indices[i].item()
    token_str = tokenizer.decode([token_id])  # Convert token ID back to string
    prob = top_probs[i].item()
    print(f"Rank {i+1}: '{token_str}' (token_id={token_id}, probability={prob:.4f})")

# Step 7: Sample one token from the full distribution (not just top-k)
# This simulates actual generation behavior with some randomness
sampled_token_id = torch.multinomial(probs, num_samples=1).item()
sampled_token = tokenizer.decode([sampled_token_id])

print(f"\nSampled token (temp={temperature}): '{sampled_token}'")
