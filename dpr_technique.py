from transformers import (
    DPRQuestionEncoder,
    DPRContextEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoderTokenizer,
)
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained DPR models and tokenizers
question_encoder = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
context_encoder = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)

# Encode a query
query = "capital of africa?"
question_inputs = question_tokenizer(query, return_tensors="pt")
question_embedding = question_encoder(**question_inputs).pooler_output

# Encode passages
passages = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
    "Rome is the capital of Italy.",
    "Maputo is the capital of Mozambique.",
    "To be or not to be, that is the question.",
    "The quick brown fox jumps over the lazy dog.",
    "Grace Hopper was an American computer scientist and United States Navy rear admiral. who was a pioneer of computer programming, and one of the first programmers of the Harvard Mark I computer. inventor of the first compiler for a computer programming language.",
]

context_embeddings = []
for passage in passages:
    context_inputs = context_tokenizer(passage, return_tensors="pt")
    context_embedding = context_encoder(**context_inputs).pooler_output
    context_embeddings.append(context_embedding)


context_embeddings = torch.cat(context_embeddings, dim=0)

# Compute similarities
similarities = cosine_similarity(
    question_embedding.detach().numpy(), context_embeddings.detach().numpy()
)
print("Similarities:", similarities)

# Get the most relevant passage
most_relevant_idx = np.argmax(similarities)
print("Most relevant passage:", passages[most_relevant_idx])
