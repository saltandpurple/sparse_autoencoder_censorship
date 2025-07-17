import umap
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import OpenAIEmbeddings
from config import *

def get_dataset_from_chromadb() -> pd.DataFrame:
    results = collection.get(
        include=['documents', 'metadatas', 'embeddings']
    )
    
    df = pd.DataFrame(results['metadatas'])
    df['embeddings'] = results['embeddings']
    df['documents'] = results['documents']
    return df

def calculate_label_distribution(df: pd.DataFrame) -> Counter:
    return Counter(df['censorship_category'].tolist())

def calculate_prompt_diversity(df: pd.DataFrame) -> Tuple[float, List[float]]:
    embeddings = np.array(df['embeddings'].tolist())
    similarity_matrix = cosine_similarity(embeddings)
    
    similarities = []
    for i in range(len(embeddings)):
        other_similarities = np.concatenate([similarity_matrix[i][:i], similarity_matrix[i][i+1:]])
        similarities.append(np.max(other_similarities))
    
    return np.mean(similarities), similarities

def generate_response_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    embed = OpenAIEmbeddings(
        model=TEXT_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )
    
    df_copy = df.copy()
    response_embeddings = []
    
    for response in df['response']:
        embedding = embed.embed_query(response)
        response_embeddings.append(embedding)
    
    df_copy['response_embeddings'] = response_embeddings
    return df_copy

def create_umap_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    embeddings = np.array(df['response_embeddings'].tolist())
    
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(embeddings)
    
    df_copy = df.copy()
    df_copy['umap_x'] = umap_coords[:, 0]
    df_copy['umap_y'] = umap_coords[:, 1]
    
    return df_copy

def calculate_response_lengths(df: pd.DataFrame) -> List[int]:
    return [len(response.split()) for response in df['response']]

def get_distinctive_ngrams(df: pd.DataFrame, n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    categories = df['censorship_category'].unique()
    distinctive_ngrams = {}
    
    for category in categories:
        category_responses = df[df['censorship_category'] == category]['response'].tolist()
        other_responses = df[df['censorship_category'] != category]['response'].tolist()
        
        category_text = ' '.join(category_responses)
        other_text = ' '.join(other_responses)
        
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1000, stop_words='english')
        
        tfidf_matrix = vectorizer.fit_transform([category_text, other_text])
        feature_names = vectorizer.get_feature_names_out()
        
        category_scores = tfidf_matrix[0].toarray()[0]
        other_scores = tfidf_matrix[1].toarray()[0]
        
        log_odds_ratios = []
        for i, feature in enumerate(feature_names):
            category_score = category_scores[i] + 1e-10
            other_score = other_scores[i] + 1e-10
            log_odds = np.log(category_score / other_score)
            log_odds_ratios.append((feature, log_odds))
        
        log_odds_ratios.sort(key=lambda x: x[1], reverse=True)
        distinctive_ngrams[category] = log_odds_ratios[:n]
    
    return distinctive_ngrams

def get_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        'total_questions': len(df),
        'total_censored': len(df[df['censored'] == True]),
        'censorship_rate': len(df[df['censored'] == True]) / len(df) * 100,
        'unique_subjects': df['subject'].nunique(),
        'date_range': {
            'earliest': df['timestamp'].min(),
            'latest': df['timestamp'].max()
        }
    }