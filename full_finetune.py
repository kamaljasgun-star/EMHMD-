import pandas as pd
import torch
import time
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------- Step 1: Load data ----------
train_df = pd.read_csv("Data/Constraint_Train.csv")
val_df = pd.read_csv("Data/Constraint_Val.csv")
test_df = pd.read_csv("Data/Constraint_Test.csv")

label_map = {"real": 1, "fake": 0}
train_df['label_num'] = train_df['label'].map(label_map)
val_df['label_num'] = val_df['label'].map(label_map)
test_df['label_num'] = test_df['label'].map(label_map)

# ---------- Step 2: Load BioBERT ----------
print("Loading BioBERT...")
model_name = "dmis-lab/biobert-base-cased-v1.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
print("BioBERT loaded!")

# ---------- Step 3: Dataset class (batching ke liye zaroori) ----------
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

train_dataset = TweetDataset(train_df['tweet'].tolist(), train_df['label_num'].tolist())
val_dataset = TweetDataset(val_df['tweet'].tolist(), val_df['label_num'].tolist())
test_dataset = TweetDataset(test_df['tweet'].tolist(), test_df['label_num'].tolist())

BATCH_SIZE = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

# ---------- Step 4: Training setup ----------
optimizer = AdamW(model.parameters(), lr=2e-5)
NUM_EPOCHS = 2

device = torch.device("cpu")
model.to(device)

# ---------- Step 5: Training loop ----------
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
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
        total_loss += loss.item()

        if batch_idx % 20 == 0:
            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | Time elapsed: {elapsed/60:.1f} min")

    avg_loss = total_loss / len(train_loader)
    epoch_time = time.time() - epoch_start
    print(f"\n=== Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time/60:.1f} min ===\n")

    torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}.pt")
    print(f"Checkpoint saved: checkpoint_epoch{epoch+1}.pt\n")

print("Training complete!")

# ---------- Step 6: Final evaluation on TEST set ----------
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
print("FINAL TEST RESULTS (Full CONSTRAINT dataset)")
print("="*50)
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision (macro): {prec:.4f}")
print(f"Recall (macro): {rec:.4f}")
print(f"F1-Score (macro): {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")

torch.save(model.state_dict(), "biobert_full_finetuned_final.pt")
print("\nFinal model saved as biobert_full_finetuned_final.pt")