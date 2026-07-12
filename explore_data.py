import pandas as pd

# Load the three splits
train_df = pd.read_csv("Data/Constraint_Train.csv")
val_df = pd.read_csv("Data/Constraint_Val.csv")
test_df = pd.read_csv("Data/Constraint_Test.csv")

# Quick check
print(f"Train: {len(train_df)} examples")
print(train_df['label'].value_counts())
print()
print(f"Validation: {len(val_df)} examples")
print(val_df['label'].value_counts())
print()
print(f"Test: {len(test_df)} examples")
print(test_df['label'].value_counts())

# Look at a few examples
print("\n--- Sample rows ---")
print(train_df.head(3))
import re

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)   # URLs hatao
    text = re.sub(r'\s+', ' ', text)       # extra spaces hatao
    return text.strip()

# Cleaning apply karo teeno datasets pe
train_df['tweet_clean'] = train_df['tweet'].apply(clean_text)
val_df['tweet_clean'] = val_df['tweet'].apply(clean_text)
test_df['tweet_clean'] = test_df['tweet'].apply(clean_text)

# Before vs After dekho
print("\n--- Cleaning Example ---")
print("BEFORE:", train_df['tweet'].iloc[1])
print("AFTER: ", train_df['tweet_clean'].iloc[1])