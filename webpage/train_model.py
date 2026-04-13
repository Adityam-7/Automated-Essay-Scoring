import nltk
import re
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textstat import flesch_reading_ease
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import torch
from syllables import estimate as syllable_count

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def load_data(file_path):
    df = pd.read_csv(file_path)  
    df = df[:15000]
    df = df.rename(columns={df.columns[0]: 'essay_id', df.columns[1]: 'score', df.columns[2]: 'essay_text'})
    df = df[['essay_text', 'score']].dropna()
    df['score'] = pd.to_numeric(df['score'], errors='coerce').dropna().astype(int)
    df = df[df['score'].isin([1, 2, 3, 4])]
    return df

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def extract_linguistic_features(essay):
    original_text = str(essay)
    sentences = sent_tokenize(original_text)
    words = word_tokenize(original_text)
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    avg_sentence_length = word_count / sentence_count
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    readability = flesch_reading_ease(original_text)
    unique_words = len(set(words))
    lexical_diversity = unique_words / word_count if word_count > 0 else 0
    syllable_count_total = sum(syllable_count(word) for word in words)
    avg_syllables_per_word = syllable_count_total / word_count if word_count > 0 else 0
    return [word_count, sentence_count, avg_sentence_length, avg_word_length, readability, lexical_diversity, avg_syllables_per_word]

def extract_tfidf_features(essays, fit=True, vectorizer=None):
    if fit:
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(essays)
        return tfidf_matrix, vectorizer
    return vectorizer.transform(essays), vectorizer

def extract_bert_embeddings(essays, model_name='bert-base-uncased', max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if isinstance(essays, str):
        essays = [essays]
    embeddings = []
    for essay in essays:
        inputs = tokenizer(essay, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_embedding)
    return np.array(embeddings)

def combine_features(linguistic_features, tfidf_features, bert_embeddings, scaler=None, fit_scaler=True):
    linguistic_array = np.array(linguistic_features)
    if fit_scaler:
        scaler = StandardScaler()
        linguistic_scaled = scaler.fit_transform(linguistic_array)
    else:
        linguistic_scaled = scaler.transform(linguistic_array)
    tfidf_array = tfidf_features.toarray() if not isinstance(tfidf_features, np.ndarray) else tfidf_features
    combined = np.hstack((linguistic_scaled, tfidf_array, bert_embeddings))
    return combined, scaler

def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y - 1, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=4, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("XGBoost Accuracy:", np.mean(y_pred == y_test))
    return model

# Main function
file_path = 'C:/Users/Aditya/OneDrive/Documents/jeevana/ASAP2_train_sourcetexts.csv'  
df = load_data(file_path)
df['cleaned_essay'] = df['essay_text'].apply(preprocess_text)
linguistic_features = df['essay_text'].apply(extract_linguistic_features).tolist()
tfidf_matrix, tfidf_vectorizer = extract_tfidf_features(df['cleaned_essay'])
bert_embeddings = extract_bert_embeddings(df['cleaned_essay'])
X, scaler = combine_features(linguistic_features, tfidf_matrix, bert_embeddings)
y = df['score'].values
model = train_xgboost(X, y)
joblib.dump(model, 'xgboost_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(scaler, 'scaler.pkl')