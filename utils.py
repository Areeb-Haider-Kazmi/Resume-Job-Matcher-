import fitz  # PyMuPDF
import torch
import nltk
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download resources safely
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load Hugging Face model & tokenizer
MODEL_NAME = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element: last hidden state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
           torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_bert_embedding(text):
    encoded = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded)
    return mean_pooling(output, encoded['attention_mask']).numpy()

def compute_similarity_bert(resume_text, jd_text):
    emb1 = get_bert_embedding(resume_text)
    emb2 = get_bert_embedding(jd_text)
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return float(similarity) * 100

def preprocess(text):
    # Lowercase, remove stopwords, simple word split
    text = text.lower()
    tokens = text.split()  # no punkt needed
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

def extract_keywords(resume, jd):
    resume_tokens = set(preprocess(resume).split())
    jd_tokens = set(preprocess(jd).split())
    return list(jd_tokens & resume_tokens)
