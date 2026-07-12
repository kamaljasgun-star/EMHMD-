import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# ---------- Step 1: Load and combine CoAID data ----------
fake_df = pd.read_csv("Data/NewsFakeCOVID-19.csv")
real_df = pd.read_csv("Data/NewsRealCOVID-19.csv")

fake_df = fake_df[['content']].dropna()
fake_df['label_num'] = 0  # fake

real_df = real_df[['content']].dropna()
real_df['label_num'] = 1  # real

combined_df = pd.concat([fake_df, real_df], ignore_index=True)
combined_df = combined_df[combined_df['content'].str.len() > 20]

print(f"Cross-dataset (CoAID) size: {len(combined_df)}")
print(f"Fake: {(combined_df['label_num']==0).sum()} | Real: {(combined_df['label_num']==1).sum()}")

# ---------- Step 2: Load tokenizer + trained model ----------
print("\nLoading trained BioBERT model...")
model_name = "dmis-lab/biobert-base-cased-v1.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.load_state_dict(torch.load("biobert_full_finetuned_final.pt", map_location="cpu"))
model.eval()
print("Model loaded successfully!")

# ---------- Step 3: Dataset class ----------
class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx])
        }

cross_dataset = TweetDataset(combined_df['content'].tolist(), combined_df['label_num'].tolist())
cross_loader = DataLoader(cross_dataset, batch_size=8)

# ---------- Step 4: Run evaluation ----------
device = torch.device("cpu")
model.to(device)

all_preds = []
all_labels = []

print("\nRunning cross-dataset evaluation (this may take a while)...")
with torch.no_grad():
    for batch in cross_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ---------- Step 5: Metrics ----------
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='macro')
rec = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
cm = confusion_matrix(all_labels, all_preds)

print("\n" + "="*50)
print("CROSS-DATASET GENERALIZATION RESULTS (CoAID)")
print("="*50)
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision (macro): {prec:.4f}")
print(f"Recall (macro): {rec:.4f}")
print(f"F1-Score (macro): {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")