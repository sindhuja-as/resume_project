from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_resume(candidate, reference):
       # ROUGE Scores
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(reference, candidate)

    # BERTScore
    P, R, F1 = bert_score([candidate], [reference], lang="en", rescale_with_baseline=True)
    


    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([candidate, reference])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    


    results = {
        "ROUGE-1": rouge_scores["rouge1"].fmeasure,
        "ROUGE-2": rouge_scores["rouge2"].fmeasure,
        "ROUGE-L": rouge_scores["rougeL"].fmeasure,
        "BERTScore-F1": F1[0].item(),
        "Cosine Similarity":round(float(cosine_sim[0][0]), 4)
    }

    return results
