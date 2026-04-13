#Importing Required Libraries needed
import nltk
import re
import numpy as np
import joblib
import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textstat import flesch_reading_ease
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
from syllables import estimate as syllable_count
#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#Downloading required packages and handling errors
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK data: {str(e)}")
    raise
#Initializing FastAPI
app = FastAPI()
#Mounting to act as server file for html
app.mount("/static", StaticFiles(directory=".", html=True), name="static")
#Loading pre-trained model and preprocessors
try:
    model = joblib.load('xgboost_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError as e:
    logger.error(f"Model file missing: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Model file missing: {str(e)}")
except Exception as e:
    logger.error(f"Failed to load model files: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Failed to load model files: {str(e)}")

class EssayRequest(BaseModel):
    essay: str
#Function to preprocess the essay
def preprocess_text(text):
    try:
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error in preprocess_text: {str(e)}")
        raise
# Extract linguistic features from the essay
def extract_linguistic_features(essay):
    try:
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
    except Exception as e:
        logger.error(f"Error in extract_linguistic_features: {str(e)}")
        raise
# Extracting TF-IDF features from text
def extract_tfidf_features(essays, fit=False, vectorizer=None):
    try:
        if fit:
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(essays)
            return tfidf_matrix, vectorizer
        return vectorizer.transform(essays), vectorizer
    except Exception as e:
        logger.error(f"Error in extract_tfidf_features: {str(e)}")
        raise
#Extract BERT embeddings for semantic representation
def extract_bert_embeddings(essays, model_name='bert-base-uncased', max_length=512):
    try:
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
    except Exception as e:
        logger.error(f"Error in extract_bert_embeddings: {str(e)}")
        raise
# Combine all the above features
def combine_features(linguistic_features, tfidf_features, bert_embeddings, scaler=None, fit_scaler=False):
    try:
        linguistic_array = np.array(linguistic_features)
        if fit_scaler:
            scaler = StandardScaler()
            linguistic_scaled = scaler.fit_transform(linguistic_array)
        else:
            linguistic_scaled = scaler.transform(linguistic_array)
        tfidf_array = tfidf_features.toarray() if not isinstance(tfidf_features, np.ndarray) else tfidf_features
        combined = np.hstack((linguistic_scaled, tfidf_array, bert_embeddings))
        return combined, scaler
    except Exception as e:
        logger.error(f"Error in combine_features: {str(e)}")
        raise
# Scoring an essay using the trained model
def score_essay(essay, model, tfidf_vectorizer, scaler):
    try:
        cleaned_essay = preprocess_text(essay)
        linguistic_features = extract_linguistic_features(essay)
        tfidf_features, _ = extract_tfidf_features([cleaned_essay], fit=False, vectorizer=tfidf_vectorizer)
        bert_embeddings = extract_bert_embeddings(cleaned_essay)
        combined_features, _ = combine_features([linguistic_features], tfidf_features, bert_embeddings, scaler=scaler, fit_scaler=False)
        score = model.predict(combined_features)[0] + 1
        return score
    except Exception as e:
        logger.error(f"Error in score_essay: {str(e)}")
        raise
# API endpoint to score essays
@app.post("/score_essay")
async def score_essay_endpoint(request: EssayRequest):
    essay = request.essay.strip()
    if not essay:
        logger.error("Empty essay provided")
        raise HTTPException(status_code=400, detail="Essay cannot be empty")
    if len(essay) > 10000:
        logger.error("Essay exceeds maximum length")
        raise HTTPException(status_code=400, detail="Essay is too long")
    
    try:
        score = score_essay(essay, model, tfidf_vectorizer, scaler)
        logger.info(f"Successfully scored essay, score: {score}")
        return {"score": int(score)}
    except Exception as e:
        logger.error(f"Error processing essay: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing essay: {str(e)}")
# Serving the index.html web page
@app.get("/")
async def serve_index():
    try:
        return FileResponse('index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        raise HTTPException(status_code=404, detail="Index page not found")
# Serve the favicon.ico file
@app.get("/favicon.ico")
async def favicon():
    try:
        return FileResponse('favicon.ico')
    except Exception as e:
        logger.error(f"Favicon not found: {str(e)}")
        raise HTTPException(status_code=404, detail="Favicon not found")