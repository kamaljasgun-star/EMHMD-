import pandas as pd
import numpy as np

# Load test data and saved predictions
test_df = pd.read_csv("Data/Constraint_Test.csv")
biobert_preds = np.load("biobert_predictions.npy")
biobert_labels = np.load("biobert_true_labels.npy")
tfidf_preds = np.load("tfidf_predictions.npy")
tfidf_labels = np.load("tfidf_true_labels.npy")

biobert_correct = (biobert_preds == biobert_labels)
tfidf_correct = (tfidf_preds == tfidf_labels)

# Find cases where BioBERT is correct but TF-IDF is wrong
only_biobert_correct = biobert_correct & ~tfidf_correct

print(f"Total cases where only BioBERT is correct: {only_biobert_correct.sum()}\n")
print("="*60)
print("TOP 5 DEMO-READY EXAMPLES (BioBERT correct, TF-IDF wrong):")
print("="*60)

indices = np.where(only_biobert_correct)[0][:5]
for i, idx in enumerate(indices):
    true_label = "REAL" if biobert_labels[idx] == 1 else "FAKE"
    tfidf_said = "REAL" if tfidf_preds[idx] == 1 else "FAKE"
    tweet_text = test_df.iloc[idx]['tweet']
    print(f"\nExample {i+1}:")
    print(f"Tweet: {tweet_text}")
    print(f"True label: {true_label}")
    print(f"TF-IDF (wrongly) said: {tfidf_said}")
    print(f"BioBERT correctly said: {true_label}")