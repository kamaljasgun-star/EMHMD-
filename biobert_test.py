# Step 1: Import the tools we need from transformers library
from transformers import AutoTokenizer, AutoModel

# Step 2: Tell Python which BioBERT model to download
model_name = "dmis-lab/biobert-base-cased-v1.2"

# Step 3: Load the tokenizer (converts text into numbers)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 4: Load the actual BioBERT model (the brain)
print("Loading BioBERT model...")
model = AutoModel.from_pretrained(model_name)

# Step 5: Test with a sample health claim
test_sentence = "Drinking lemon water cures cancer"

# Step 6: Convert sentence to numbers (tokens)
tokens = tokenizer(test_sentence, return_tensors="pt")

# Step 7: Pass tokens through BioBERT to get embeddings
output = model(**tokens)

# Step 8: Print the shape to confirm it worked
print("Success! Output shape:", output.last_hidden_state.shape)