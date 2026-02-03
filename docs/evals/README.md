As you've noticed, there are quite a few hyperparameters such as the choice of embedding model, top-K, etc. that needs tuning. Here are some questions RAG evaluation aims to solve in the retrieval step:

- Does the embedding model you're using capture `domain-specific nuances`? (If you're working on a medical use case, a generic embedding model offered by OpenAI might not provide expected the vector search results.)
- Does your `reranker model` ranks the retrieved nodes in the "correct" order?
- Are you retrieving the right amount of information? This is influenced by hyperparameters text chunk size, top-K number.

`ContextualPrecisionMetric`: evaluates whether the reranker in your retriever ranks more relevant nodes in your retrieval context higher than irrelevant ones.

`ContextualRecallMetric`: evaluates whether the embedding model in your retriever is able to accurately capture and retrieve relevant information based on the context of the input.

`ContextualRelevancyMetric`: evaluates whether the text chunk size and top-K of your retriever is able to retrieve information without much irrelevancies.
