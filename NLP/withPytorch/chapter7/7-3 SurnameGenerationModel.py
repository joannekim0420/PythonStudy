import torch
import torch.nn as nn
import torch.nn.functional as F

class SurnameGenerationModel(nn.Module):
    def __init(self, char_embedding_size, char_vocab_size, rnn_hidden_size, batch_first = True, padding_idx =0, dropout_p = 0.5):
        super().__init__()

        self.char_emb = nn.Embedding(num_emb)