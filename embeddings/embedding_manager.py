import numpy as np

class EmbeddingManager:
    EMBEDDINGS_FILE = "embeddings/datastories.twitter.300d.txt"

    def get_embedding_matrix(self, word_index, embedding_dim):
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        embeddings_index = self.get_embeddings()

        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def get_embeddings(self):
        embeddings_index = {}
        with open(self.EMBEDDINGS_FILE, encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(embeddings_index))
        return embeddings_index
