import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

# Load predictions and true labels
biobert_preds = np.load("biobert_predictions.npy")
biobert_labels = np.load("biobert_true_labels.npy")
tfidf_preds = np.load("tfidf_predictions.npy")
tfidf_labels = np.load("tfidf_true_labels.npy")

print("BioBERT test set size:", len(biobert_preds))
print("TF-IDF test set size:", len(tfidf_preds))

# Check correctness of each model's predictions
biobert_correct = (biobert_preds == biobert_labels)
tfidf_correct = (tfidf_preds == tfidf_labels)

# Build 2x2 contingency table
both_correct = np.sum(biobert_correct & tfidf_correct)
only_biobert_correct = np.sum(biobert_correct & ~tfidf_correct)
only_tfidf_correct = np.sum(~biobert_correct & tfidf_correct)
both_wrong = np.sum(~biobert_correct & ~tfidf_correct)

print("\nContingency Table:")
print(f"Both correct: {both_correct}")
print(f"Only BioBERT correct: {only_biobert_correct}")
print(f"Only TF-IDF correct: {only_tfidf_correct}")
print(f"Both wrong: {both_wrong}")

table = [[both_correct, only_biobert_correct],
         [only_tfidf_correct, both_wrong]]

result = mcnemar(table, exact=True)
print(f"\nMcNemar's test statistic: {result.statistic}")
print(f"p-value: {result.pvalue}")

if result.pvalue < 0.05:
    print("\nResult: Statistically significant difference (p < 0.05)")
    print("BioBERT's improvement over TF-IDF is NOT due to random chance.")
else:
    print("\nResult: No statistically significant difference (p >= 0.05)")