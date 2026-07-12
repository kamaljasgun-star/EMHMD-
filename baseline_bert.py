import pandas as pd
import torch
import time
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

train_df = pd.read_csv("Data/Constraint_Train.csv")
test_df = pd.read_csv("Data/Constraint_Test.csv")

label_map = {"real": 1, "fake": 0}
train_df['label_num'] = train_df['label'].map(label_map)
test_df['label_num'] = test_df['label'].map(label_map)

print("Loading general BERT (bert-base-cased)...")
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
print("BERT loaded!")

class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = tokenizer(str(self.texts[idx]), truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(0), 'attention_mask': encoding['attention_mask'].squeeze(0), 'label': torch.tensor(self.labels[idx])}

train_dataset = TweetDataset(train_df['tweet'].tolist(), train_df['label_num'].tolist())
test_dataset = TweetDataset(test_df['tweet'].tolist(), test_df['label_num'].tolist())

BATCH_SIZE = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

optimizer = AdamW(model.parameters(), lr=2e-5)
NUM_EPOCHS = 2
device = torch.device("cpu")
model.to(device)

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_start = time.time()
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Time: {elapsed/60:.1f} min")
    print(f"=== Epoch {epoch+1} complete ===")

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='macro')
rec = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
cm = confusion_matrix(all_labels, all_preds)

print("\n" + "="*50)
print("GENERAL BERT (baseline) RESULTS")
print("="*50)
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision (macro): {prec:.4f}")
print(f"Recall (macro): {rec:.4f}")
print(f"F1-Score (macro): {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")

np.save("bert_predictions.npy", np.array(all_preds))
print("\nSaved predictions for statistical test.")