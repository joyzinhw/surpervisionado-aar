import numpy as np

from scipy.spatial.distance import cosine
from scipy import spatial


def cosine_similarity(snt1, snt2):
    return 1 - cosine(snt1, snt2)


def cosine_distance_embeddings(vector, tokens1, tokens2):
    def get_mean_vec(tokens):
        vecs = [vector[token] for token in tokens if token in vector.key_to_index]
        if not vecs:
            return np.zeros(vector.vector_size)  # shape = (300,)
        return np.mean(vecs, axis=0).ravel()

    vector1 = get_mean_vec(tokens1)
    vector2 = get_mean_vec(tokens2)

    # Evita erro caso ambos sejam vetores nulos
    if np.all(vector1 == 0) or np.all(vector2 == 0):
        return 0.0

    return 1 - spatial.distance.cosine(vector1, vector2)