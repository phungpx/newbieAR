import numpy as np
import openai

model_id = "text-embedding-all-minilm-l6-v2-embedding"
base_url = "http://127.0.0.1:1234/v1"
api_key = "empty"

client = openai.OpenAI(base_url=base_url, api_key=api_key)

# query_prefix = "query: "
query_prefix = ""
queries = [
    "What did Napoléon achieve?",
    # "what is snowflake?",
    # "Where can I get the best tacos?",
    # "What is VNS Company? What is the company's mission?",
]
queries_with_prefix = [f"{query_prefix}{query}" for query in queries]

documents = [
    "Napoléon became Emperor in 1804",
    "Napoléon was born in Corsica on August 15, 1769",
    "He became Emperor in 1804",
    "Napoléon introduced the Napoleonic Code, which reformed the French legal system and influenced civil law around the world",
    "Napoléon reorganized the French administrative and educational systems, helping to lay the foundations of the modern French state",
    "Napoléon was defeated at the Battle of Waterloo in 1815",
    "Napoléon spent his final years in exile on the island of Saint Helena",
    # "The Data Cloud!",
    # "Mexico City of Course!",
    # "VNS Company is a software company. The company's mission is to provide software solutions to businesses.",
]

query_embeddings = client.embeddings.create(model=model_id, input=queries_with_prefix)
document_embeddings = client.embeddings.create(model=model_id, input=documents)

# Convert embeddings to numpy arrays
query_embed_array = np.array([data.embedding for data in query_embeddings.data])
document_embed_array = np.array([data.embedding for data in document_embeddings.data])

# Normalize embeddings (L2 normalization)
query_embed_array = query_embed_array / np.linalg.norm(
    query_embed_array, ord=2, axis=1, keepdims=True
)
document_embed_array = document_embed_array / np.linalg.norm(
    document_embed_array, ord=2, axis=1, keepdims=True
)

# Compute cosine similarity scores
scores = np.dot(query_embed_array, document_embed_array.T)

# Output results
for query, query_scores in zip(queries_with_prefix, scores):
    print("-" * 20)
    doc_score_pairs = list(zip(documents, query_scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    print("Input:", query)
    for document, score in doc_score_pairs:
        print(f"- ({score:.2f}) {document}")
