import numpy as np
from annoy import AnnoyIndex

class PreTrainedEmbeddings(object):
    def __init__(self, word_to_index, word_vectors):
        self.word_to_index = word_to_index
        self.word_vectors = word_vectors
        self.index_to_word = {v:k for k,v in self.word_to_index.items()}
        self.index = AnnoyIndex(len(word_vectors[0]), metric="euclidean")

        for _, i in self.word_to_index.items():
            self.index.add_item(i, self.word_vectores[i])
        self.index.build(50)

    @classmethod
    def from_embeddings_file(cls, embedding_file):
        word_to_index = {}
        word_vectors = []
        """
        vector files are like:
        word0 x0_0 x0_1 x0_2 ...
        word1 x1_0 x1_1 x1_2 ...
        """
        with open(embedding_file) as fp:
            for line in fp.readlines():
                line = line.split(" ")
                word = line[0]
                vec = np.array([float(x) for x in line[1:]])

                word_to_index[word] = len(word_to_index)
                word_vectors.append(vec)
        return cls(word_to_index, word_vectors)

    def get_embedding(self, word):
        return self.word_vectors[self.word_to_index[word]]

    def get_closest_to_vector(self, vector, n=1):
        nn_indices = self.index.get_nns_by_vector(vector, n)
        return [self.index_to_word[neighbor] for neighbor in nn_indices]

    def compute_and_print_analogy(self, word1, word2, word3):
        """
        word1: word2 :: word3 : word4
        """
        vec1 = self.get_embedding(word1)
        vec2 = self.get_embedding(word2)
        vec3 = self.get_embedding(word3)

        spatial_relationship = vec2-vec1
        vec4 = vec3 + spatial_relationship

        closest_words = self.get_closest_to_vector(vec4, n=4)
        existing_word = set([word1, word2, word3])
        closest_words = [word for word in closest_words if word not in existing_word]

        if len(closest_words) ==0:
            print("can't find closest word")
            return
        for word4 in closest_words:
            print("{} : {} :: {} : {}".format(word1, word2, word3, word4))

        ###OUTPUT EXAMPLE###
        #embeddings.compute_and_print_analogy('man','he','woman')
        #man : he :: woman : she

embeddings = PreTrainedEmbeddings.from_embeddings_file('GoogleNews-vectors-negative300.bin')