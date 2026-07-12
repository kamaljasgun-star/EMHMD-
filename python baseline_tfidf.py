import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

train_df = pd.read_csv("Data/Constraint_Train.csv")
test_df = pd.read_csv("Data/Constraint_Test.csv")

label_map = {"real": 1, "fake": 0}
train_df['label_num'] = train_df['label'].map(label_map)
test_df['label_num'] = test_df['label'].map(label_map)

print("Converting text to TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(train_df['tweet'])
X_test = vectorizer.transform(test_df['tweet'])

y_train = train_df['label_num']
y_test = test_df['label_num']

print("Training Logistic Regression...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, average='macro')
rec = recall_score(y_test, preds, average='macro')
f1 = f1_score(y_test, preds, average='macro')
cm = confusion_matrix(y_test, preds)

print("\n" + "="*50)
print("TF-IDF + LOGISTIC REGRESSION RESULTS")
print("="*50)
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision (macro): {prec:.4f}")
print(f"Recall (macro): {rec:.4f}")
print(f"F1-Score (macro): {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")

np.save("tfidf_predictions.npy", preds)
np.save("tfidf_true_labels.npy", y_test.values)
print("\nSaved predictions for later statistical test.")