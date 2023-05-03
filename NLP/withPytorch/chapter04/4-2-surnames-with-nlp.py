import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SurnameDataset(Dataset):
    def __init__(self, surname_df, vectorizer):
        self.surname_df = surname_df
        self._vectorizer = vectorizer

        self.train_df = self.surname_df[self.surname_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.surname_df[self.surname_df.split == 'val']
        self.validation_size = len(self.val_df)

        self.test_df = self.surname_df[self.surname_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

        # 클래스 가중치
        class_counts = surname_df.nationality.value_counts().to_dict()

        def sort_key(item):
            return self._vectorizer.nationality_vocab.lookup_token(item[0])

        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        surname_vector = self._vectorizer.vectorize(row.surname)
        nationality_index = self._vecotrizer.nationality_vocab.lookip_token(row.nationality)

        return {'x_surname':surname_vector,
                'y_nationality':nationality_index}

class Vocabulary(object):
    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK"):
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token ={idx : token for token, idx in self._token_to_idx.items()}
        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self._unk_token = self.add_token(unk_token)

    def lookup_token(self, token):
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

class SurnameVectorizer(object):
    def __init__(self, surname_vocab, nationality_vocab):
        self.surname_vocab = surname_vocab
        self.nationality_vocab = nationality_vocab

    def vectorize(self, surname):
        vocab = self.surname_vocab
        one_hot = np.zeros(len(vocab),dtype=np.float32)
        for token in surname:
            one_hot[vocab.lookup_token(token)] = 1
        return one_hot

    def from_dataframe(cls, surname_df):
        surname_vocab = Vocabulary(unk_token="@")
        nationality_vocab = Vocabulary(add_unk=False)

        for index, row in surname_df.iterrows():
            for letter in row.surname:
                surname_vocab.add_token(letter)
            nationality_vocab.add_token(row.nationality)
        return cls(surname_vocab, nationality_vocab)

class SurnameClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SurnameClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x_in, apply_softmax = False):
            layer = F.relu(self.fc1(x_in))
            out_layer = self.fc2(F.dropout(layer, p=0.2))

            if apply_softmax:
                output = F.softmax(out_layer, dim=1)
            return output

args = Namespace(
    surname_csv = "datapath.csv",
    vectorizer_file = "vectorizer.json",
    model_state_File = "model.pth",
    save_dir = "model_save/path",
    hidden_dim = 300,
    seed=1337,
    num_epochs = 100,
    early_stopping_criteria=5,
    learning_rate = 0.001,
    batch_size = 64
)