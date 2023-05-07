import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

class SurnameDataset(Dataset):
    @calssmethod
    def load_dataset_and_make_vectorizer(cls, surname_csv):
        surname_df = pd.read_csv(surname_csv)
        train_surname_df = surname_df[surname_df.split=="train"]
        return cls(surname_df, SurnameVectorizer.from_dataframe(train_surname_df))

    def __getitem__(self, index):
        row = self._target_df.iloc[index]

        surname_vector, vec_length = self._vectorizer.vectorize(row.surname, self._max_seq_length)

        nationality_index = self._vectorizer.nationality_vocab.lookup_token(row.nationality)

        return {'x_data':surname_vector,
                'y_target':nationality_index,
                'x_legnth':vec_length}


class SurnameVectorizer(object):
    def vectorize(self, surname, vector_length=-1):

        indices = [self.char_vocab.begin_seq_index]
        indices.extend(self.char_vocab.lookup_token(token) for token in surname)
        indices.append(self.char_vocab.end_seq_index)

        if vector_length <0:
            vector_legnth = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.char_vocab.mask_index

        return out_vector, len(indices)

    @classmethod
    def from_dataframe(cls, surname_df):
        char_vocab = SequenceVocabulary()
        nationality_vocab = Vocabulary()

        for index, row in surname_df.iterrows():
            for char in row.surname:
                char_vocab.add_token(char)
            nationality_vocab.add_token(row.nationality)

        return cls(char_vocab, nationality_vocab)