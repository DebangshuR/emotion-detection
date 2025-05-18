# Import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from neattext import TextCleaner
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump, load
import warnings
from collections import Counter

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Function to find text and emotion columns
def find_columns(df):
    text_cols = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower() or 'sentence' in col.lower()]
    emotion_cols = [col for col in df.columns if 'emotion' in col.lower() or 'sentiment' in col.lower() or 'label' in col.lower()]
    
    if not text_cols:
        raise ValueError("Could not identify text column in CSV")
    if not emotion_cols:
        raise ValueError("Could not identify emotion column in CSV")
    
    return text_cols[0], emotion_cols[0]

# Load and prepare datasets
try:
    train_df = pd.read_csv('/content/modified_test.csv')
    test_df = pd.read_csv('/content/modified_train.csv')
    val_df = pd.read_csv('/content/modified_val.csv')
    
    # Find the correct columns
    text_col, emotion_col = find_columns(train_df)
    
    # Rename columns for consistency
    for df in [train_df, test_df, val_df]:
        df.rename(columns={text_col: 'text', emotion_col: 'emotion'}, inplace=True)
    
    # Combine all datasets for analysis
    df = pd.concat([train_df, test_df, val_df], axis=0)
    
except FileNotFoundError:
    print("CSV files not found. Using sample data instead.")
    # Fallback to sample data
    data = {
        'text': [
            "I am so happy today!", 
            "This makes me very angry", 
            "I feel sad about what happened",
            "What a wonderful day!", 
            "I'm furious with this situation",
            "This news depressed me",
            "I'm thrilled with the results",
            "I hate when this happens",
            "Feeling joyful after the good news",
            "This is so frustrating",
            "I'm delighted to see you",
            "That was terrifying",
            "I'm over the moon with happiness",
            "This is absolutely disgusting",
            "I feel so lonely right now"
        ],
        'emotion': [
            'happy', 'angry', 'sad', 
            'happy', 'angry', 'sad',
            'happy', 'angry', 'happy',
            'angry', 'happy', 'angry',
            'happy', 'angry', 'sad'
        ]
    }
    df = pd.DataFrame(data)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

# Dataset Analysis
print("\n=== Dataset Analysis ===")
print(f"Total samples: {len(df)}")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# Emotion distribution
print("\nEmotion Distribution:")
print(df['emotion'].value_counts())

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='emotion')
plt.title('Emotion Distribution')
plt.xticks(rotation=45)
plt.show()

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    cleaner = TextCleaner(text)
    cleaner.remove_emojis()
    cleaner.remove_emails()
    cleaner.remove_urls()
    cleaner.remove_numbers()
    cleaner.remove_special_characters()
    cleaner.remove_puncts()
    cleaned_text = cleaner.text.lower().strip()
    return cleaned_text

# Apply text cleaning
print("\nCleaning text data...")
for df in [train_df, val_df, test_df]:
    df['cleaned_text'] = df['text'].apply(clean_text)

# Feature Extraction
print("\nCreating TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = tfidf.fit_transform(train_df['cleaned_text'])
X_val = tfidf.transform(val_df['cleaned_text'])
X_test = tfidf.transform(test_df['cleaned_text'])

y_train = train_df['emotion']
y_val = val_df['emotion']
y_test = test_df['emotion']

# Model Training
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear', probability=True, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced')
}

best_model = None
best_accuracy = 0

print("\n=== Model Evaluation ===")
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"{name} Validation Accuracy: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# Final evaluation
print(f"\nBest Model: {best_model_name}")
y_pred = best_model.predict(X_test)
print("\nTest Set Performance:")
print(classification_report(y_test, y_pred))

# Save models
print("\nSaving model artifacts...")
dump(best_model, 'emotion_detection_model.joblib')
dump(tfidf, 'tfidf_vectorizer.joblib')

# Prediction function
def predict_emotion(text, model=None, vectorizer=None):
    if not model:
        try:
            model = load('emotion_detection_model.joblib')
            vectorizer = load('tfidf_vectorizer.joblib')
        except:
            raise ValueError("Model files not found. Please train the model first.")
    
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return "neutral", {"neutral": 1.0}
    
    text_vector = vectorizer.transform([cleaned_text]).toarray()
    prediction = model.predict(text_vector)
    probabilities = model.predict_proba(text_vector)
    confidence_scores = {model.classes_[i]: probabilities[0][i] for i in range(len(model.classes_))}
    
    return prediction[0], confidence_scores

# Demo predictions
test_texts = [
    "I'm feeling great today!",
    "This situation makes me so mad",
    "I'm really upset about the news",
    "What a beautiful morning!",
    "I don't know how to feel about this",
    "",
    "This is the worst day of my life",
    "I'm so excited for the party tonight!"
]

print("\n=== Demo Predictions ===")
for text in test_texts:
    try:
        emotion, confidence = predict_emotion(text)
        print(f"\nText: '{text}'")
        print(f"Predicted Emotion: {emotion}")
        print("Confidence Scores:")
        for emo, score in sorted(confidence.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emo}: {score:.4f}")
    except Exception as e:
        print(f"\nError processing text: '{text}'")
        print(f"Error: {str(e)}")
