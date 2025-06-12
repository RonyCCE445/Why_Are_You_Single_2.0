import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# --------- Helper function for text cleaning ---------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'\|\|\|', ' ', text)  # replace post separators with space
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace
    return text

# --------- Load and preprocess dataset ---------
df = pd.read_csv('../data/mbti_1.csv')

print(f"Original dataset size: {len(df)}")

df['posts'] = df['posts'].apply(clean_text)

# Optional: limit dataset size if needed for faster training
# df = df.sample(5000, random_state=42)

X = df['posts'].values
y = df['type'].values

# --------- Compute class weights ---------
classes = np.unique(y)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, class_weights))
print("Class weights:", class_weight_dict)

# --------- Setup model pipeline ---------
from sklearn.svm import LinearSVC

# Inside your pipeline:
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
    ('clf', LinearSVC(class_weight=class_weight_dict, max_iter=10000, dual='auto')),
])


# --------- Cross-validation for evaluation ---------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    accuracies.append(acc)

    print(f"Fold {fold} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, zero_division=0))

print(f"\nAverage CV Accuracy: {np.mean(accuracies):.4f}")

# --------- Train final model on full dataset ---------
pipeline.fit(X, y)

# --------- Save model and vectorizer ---------
joblib.dump(pipeline, 'mbti_pipeline.pkl')


print("Model and vectorizer saved!")
