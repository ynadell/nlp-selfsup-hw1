import pickle
import faiss
import numpy as np
from datasets import load_dataset

corpus = load_dataset("scifact", "corpus")
claims = load_dataset("scifact", "claims")
print("Claims Example:")
print(claims['train'][0])

ground_truth = {}

from datasets import load_dataset

corpus = load_dataset("scifact", "corpus")
claims = load_dataset("scifact", "claims")

ground_truth = {}

for example in claims['train']:
    claim_id = example['id'] 
    relevant_docs = set(example['cited_doc_ids'])

    ground_truth[claim_id] = relevant_docs


evidence_file = "scifact_evidence_embeddings.pkl"
claims_file = "scifact_claim_embeddings.pkl"

with open(evidence_file, "rb") as f:
    evidence_embeddings = pickle.load(f)

with open(claims_file, "rb") as f:
    claim_embeddings = pickle.load(f)

evidence_ids = list(evidence_embeddings.keys())
evidence_vectors = np.array([embedding for embedding in evidence_embeddings.values()])
print(evidence_vectors.shape)
dimension = evidence_vectors.shape[1]
print(dimension)
index = faiss.IndexFlatL2(dimension) 
print(index.is_trained)
index.add(evidence_vectors)          
print(index.ntotal)

def evaluate_faiss_accuracy(claim_embeddings, evidence_embeddings, ground_truth, k=5):
    total_precision = 0.0
    total_queries = len(claim_embeddings)
    mrr_score = 0.0

    def average_precision(retrieved_docs, relevant_docs):
        relevant_set = set(relevant_docs)
        retrieved_set = set(retrieved_docs)
        if not relevant_set:
            return 0.0
        precision_sum = 0.0
        relevant_count = 0
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_set:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        return precision_sum / len(relevant_docs)

    for claim, claim_embedding in claim_embeddings.items():
        claim_id, claim_text = claim
        claim_vector = np.array([claim_embedding])
        distances, indices = index.search(claim_vector, k)
        retrieved_evidence_ids = [evidence_ids[idx][0] for idx in indices[0]]
        relevant_docs = ground_truth.get(claim_id, set())

        avg_prec = average_precision(retrieved_evidence_ids, relevant_docs)
        total_precision += avg_prec

        for rank, retrieved_id in enumerate(retrieved_evidence_ids):
            if retrieved_id in relevant_docs:
                mrr_score += 1 / (rank + 1)

    map_score = total_precision / total_queries
    mrr = mrr_score / total_queries

    print(f"Mean Average Precision (MAP): {map_score:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

evaluate_faiss_accuracy(claim_embeddings, evidence_embeddings, ground_truth, k=5)
