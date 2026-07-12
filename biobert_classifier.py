# ============================================
# BioBERT Classification Test - Kamaljeet
# Purpose: Test if BioBERT can give Fake/Real prediction
# ============================================

# Import 1: Transformers - for BioBERT model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import 2: PyTorch - for math operations
import torch


# ============================================
# SECTION 2: Load Model & Tokenizer
# ============================================

# Step 1: Specify which BioBERT model to use
model_name = "dmis-lab/biobert-base-cased-v1.2"

# Step 2: Load the tokenizer (same as before)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded successfully!")

# Step 3: Load CLASSIFICATION model (different from last time!)
print("\nLoading BioBERT classifier...")
print("Note: This will create a classifier with random weights")
print("(Not yet trained on fake/real data)")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # 2 classes: Fake (0) and Real (1)
)
print("Model loaded successfully!")


# ============================================
# SECTION 3: Make a Prediction
# ============================================

# Step 1: Test sentence (health claim)
test_sentence = "Regular exercise supports heart health"
print(f"\nTest sentence: '{test_sentence}'")

# Step 2: Convert sentence to tokens
print("Tokenizing sentence...")
tokens = tokenizer(test_sentence, return_tensors="pt")

# Step 3: Get model prediction (no training mode)
print("Making prediction...")
with torch.no_grad():
    output = model(**tokens)

# Step 4: Get logits (raw scores)
logits = output.logits
print(f"\nLogits (raw scores): {logits}")

# Step 5: Convert logits to probabilities using softmax
probabilities = torch.softmax(logits, dim=1)
print(f"Probabilities: {probabilities}")

# Step 6: Extract probability values
fake_prob = probabilities[0][0].item()
real_prob = probabilities[0][1].item()

# Step 7: Print results
print("\n" + "="*50)
print("PREDICTION RESULTS:")
print("="*50)
print(f"Fake probability: {fake_prob*100:.2f}%")
print(f"Real probability: {real_prob*100:.2f}%")

# Step 8: Determine final prediction
if fake_prob > real_prob:
    prediction = "FAKE"
else:
    prediction = "REAL"

print(f"\nFinal Prediction: {prediction}")
print("="*50)

# Important note
print("\nNOTE: Since classifier is untrained, predictions are random!")
print("After fine-tuning on ANTi-Vax dataset, predictions will be accurate.")