class Vocabs:
    def __init__(self):
        self.Id2Vocab = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.Vocab2Id = {token: int for int, token in self.Id2Vocab.items()}

    def __len__(self):
        return len(self.vocabDict)

    @staticmethod
    def tokenizer(text):
        return [t.lower().strip() for t in text.split(" ")]

    def build_vocabulary(self, meta):
        idx = 4
        for word in meta['tokens']:
            self.Vocab2Id[word] = idx
            self.Id2Vocab[idx] = word
            idx += 1

    def Encode(self, sentence):
        encoded_text = []
        tokenized_text = self.tokenizer(sentence)

        for token in tokenized_text:
            if token in self.Vocab2Id.keys():
                encoded_text.append(self.Vocab2Id[token])  # encoding(mapping) words to int
            else:
                encoded_text.append(self.Vocab2Id['UNK'])
        return encoded_text
