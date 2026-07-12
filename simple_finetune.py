# ============================================
# SIMPLE FINE-TUNING - Kamaljeet
# Purpose: Train BioBERT to classify Fake/Real health claims
# ============================================

# Import 1: PyTorch for deep learning
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Import 2: Transformers for BioBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import 3: Sklearn for train/test split and metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import 4: Numpy for arrays
import numpy as np

print("All libraries imported successfully!") 
# ============================================
# SECTION 2: SMALL DATASET
# 24 examples: 12 Fake + 12 Real
# ============================================

# Fake health claims (Label = 0)
fake_texts = [
    "Drinking lemon water cures cancer completely",
    "Vaccines cause autism in children",
    "Essential oils can cure diabetes naturally",
    "Garlic can replace all antibiotics for infections",
    "Homeopathy is scientifically proven to cure cancer",
    "Drinking bleach kills coronavirus in the body",
    "5G radiation causes COVID-19 infections",
    "Apple cider vinegar cures all diseases",
    "Ginger boiled in water eliminates all viruses",
    "Turmeric alone can replace chemotherapy",
    "Chugging hydrogen peroxide cleans your blood",
    "Detox teas remove toxins better than the liver",
]

# Real health claims (Label = 1)
real_texts = [
    "Regular exercise improves cardiovascular health",
    "Vaccines are effective at preventing infectious diseases",
    "A balanced diet supports overall wellbeing",
    "Antibiotics are prescribed for bacterial infections",
    "Handwashing helps prevent the spread of germs",
    "Sleep is important for immune system function",
    "Smoking increases the risk of lung cancer",
    "Blood pressure should be monitored regularly",
    "Drinking enough water is important for hydration",
    "Mental health is as important as physical health",
    "Regular check-ups help detect diseases early",
    "Sunscreen helps protect the skin from UV damage",
]

# Combine data
all_texts = fake_texts + real_texts
all_labels = [0] * len(fake_texts) + [1] * len(real_texts)

# Print info
print(f"Total examples: {len(all_texts)}")
print(f"Fake examples: {len(fake_texts)}")
print(f"Real examples: {len(real_texts)}")     
# ============================================
# SECTION 3: MODEL LOAD + TRAINING
# ============================================

# Load BioBERT model and tokenizer
print("\nLoading BioBERT...")
model_name = "dmis-lab/biobert-base-cased-v1.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
print("BioBERT loaded!")

# Split data: 80% train, 20% test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)
print(f"\nTrain: {len(train_texts)} | Test: {len(test_texts)}")

# Tokenize training data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
train_labels_tensor = torch.tensor(train_labels)

# Setup optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop - 3 epochs
print("\nStarting training...")
model.train()
for epoch in range(3):
    optimizer.zero_grad()
    outputs = model(**train_encodings, labels=train_labels_tensor)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/3 - Loss: {loss.item():.4f}")
print("\nTraining complete!")   
# ============================================
# SECTION 4: EVALUATION + METRICS
# ============================================

print("\n" + "="*50)
print("EVALUATION ON TEST DATA")
print("="*50)

# Tokenize test data
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

# Predict on test data
model.eval()
with torch.no_grad():
    outputs = model(**test_encodings)
    predictions = torch.argmax(outputs.logits, dim=-1).numpy()

# Calculate metrics
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions, average='macro', zero_division=0)
recall = recall_score(test_labels, predictions, average='macro', zero_division=0)
f1 = f1_score(test_labels, predictions, average='macro', zero_division=0)
cm = confusion_matrix(test_labels, predictions)

# Print results
print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

print(f"\nConfusion Matrix:")
print(f"                Predicted Fake  Predicted Real")
print(f"Actual Fake     {cm[0][0]}               {cm[0][1]}")
print(f"Actual Real     {cm[1][0]}               {cm[1][1]}")

# Test with new sentences
print("\n" + "="*50)
print("TESTING WITH NEW SENTENCES")
print("="*50)

new_sentences = [
    "Drinking hot lemon water cures COVID-19 completely",
    "Regular walking helps improve heart health",
]

for text in new_sentences:
    encoding = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoding)
        probs = torch.softmax(output.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        label = "REAL" if pred == 1 else "FAKE"
        confidence = probs[0][pred].item() * 100
    print(f"\nText: {text}")
    print(f"Prediction: {label} ({confidence:.2f}% confident)")

print("\n" + "="*50)
print("DONE!")
print("="*50)