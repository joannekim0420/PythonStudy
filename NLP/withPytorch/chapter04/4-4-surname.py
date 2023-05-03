import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SurnameDataset(Dataset):
    def __getitem__(self,index):
        row = self._target_df.iloc[index]
        surname_matrix = self._vectorizer.vectorize(Row.surname, self._man_seq_length)
        nationality_index = self._vectorizer.nationality_vocab.lookup_token(row.nationality)

        return {'x_surname':surname_matrix,
                'y_nationality':nationality_index}

class SurnameVectorizer(object):
    def vectorize(self, surname):
        one_hot_matrix_size = (len(self.character_vocab), self.max_surname_legnth)
        one_hot_matrix = np.zeros(one_hot_matrix_size, dtype=np.float32)
        for position_index, character in enumerate(surname):
            character_index = self.character_vocab.lookup_token(character)
            one_hot_matrix[character_index][position_index] = 1

        return one_hot_matrix

    @classmethod
    def from_dataframe(cls, surname_df):
        character_vocab = Vocabulary(unk_token="@")
        nationality_voca = Vocabulary(add_unk=False)

        max_surname_legnth = 0
        for index, row in surname_df.iterrows():
            max_surname_legnth = max(max_surname_legnth, len(row.surname))
            for letter in row.surname:
                character_vocab.add_token(letter)
            nationality_vocab.add_token(row.nationality)

        return clas(character_vocab, nationality_vocab, max_surname_legnth)

class SurnameClassifier(nn.Module):
    def __init__(self, initial_num_channels, num_classes, num_channels):
        super(SurnameClassifier, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=initial_num_channels, out_channels=num_channels, kernel_size=3),
            nn.BatchNorm1d(num_features=num_channels)
            nn.ELU(),

            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3),
            nn.ELU(),

            nn.Conv1d(in_channels=num_channels,out_channels=num_channels, kernel_size=3),
            nn.ELU(),

            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3),
            nn.ELU()
        )

        self.fc = nn.Linear(num_channels,num_classes)

    def forward(self, x_surname, apply_softmax=False):
        features = self.convnet(x_surname).squeeze(dim=2)
        prediction_vector = self.fc(features)

        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector)

        return prediction_vector