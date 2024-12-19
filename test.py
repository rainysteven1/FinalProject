from gensim.models import KeyedVectors
import numpy as np
import pandas as pd

model = KeyedVectors.load_word2vec_format(
    "./resources/GoogleNews-vectors-negative300.bin", binary=True
)


def build_vector(text, model):
    vec = np.zeros(model.vector_size).reshape((1, model.vector_size))
    count = 0

    for word in text:
        try:
            vec += model[word].reshape((1, model.vector_size))
            count += 1
        except KeyError:
            continue

    if count != 0:
        vec /= count

    return vec


chunks = list()
for chunk in pd.read_csv("./resources/train.csv", chunksize=100000):
    chunk["vec"] = chunk["content"].apply(lambda x: build_vector(x, model))
    chunks.append(chunk)

df = pd.concat(chunks)
df.to_csv("test.csv", index=False)
