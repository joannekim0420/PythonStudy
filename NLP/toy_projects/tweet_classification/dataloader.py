import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from vocabulary import Vocabs
from torch.nn.utils.rnn import pad_sequence

train_data = json.load(open("./data/train.json", 'r'))
valid_data = json.load(open("./data/valid.json", 'r'))
test_data = json.load(open("./data/test.json", 'r'))
meta_data = json.load(open("./meta.json", 'r'))

pad_idx = 0


class CustomDataset(Dataset):
    def __init__(self, dataDict):
        self.dataDict = dataDict
        self.vocabs = Vocabs()
        self.vocabs.build_vocabulary(meta_data)

    def __len__(self):
        return len(self.dataDict)

    def __getitem__(self, idx):
        current_sentence = self.dataDict[idx]['sentence']
        sentence = [self.vocabs.Vocab2Id["<SOS>"]]
        sentence += self.vocabs.Encode((current_sentence))
        sentence.append(self.vocabs.Vocab2Id["<EOS>"])

        user_id = self.dataDict[idx]['user_id']
        label = torch.LongTensor([user_id])
        return torch.tensor(sentence), label


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        text = [item[0] for item in batch]

        text = pad_sequence(text, batch_first=True, padding_value=self.pad_idx)
        labels = [item[1] for item in batch]
        labels = torch.LongTensor(labels)
        return text, labels


def load_data(config, mode):
    train_dataset = CustomDataset(train_data)
    valid_dataset = CustomDataset(train_data)
    test_dataset = CustomDataset(train_data)

    if mode == "train":
        train_loader = DataLoader(train_dataset, batch_size=config['Train']['batch_size'], shuffle=True, num_workers=0,
                                  collate_fn=MyCollate(pad_idx=pad_idx))
        valid_loader = DataLoader(valid_dataset, batch_size=config['Train']['batch_size'], shuffle=True, num_workers=0,
                                  collate_fn=MyCollate(pad_idx=pad_idx))
        return train_loader, valid_loader, len(train_dataset.vocabs.Id2Vocab)

    else:
        test_loader = DataLoader(test_dataset, batch_size=config['Test']['batch_size'], shuffle=True, num_workers=0,
                                 collate_fn=MyCollate(pad_idx=pad_idx))
        return test_loader