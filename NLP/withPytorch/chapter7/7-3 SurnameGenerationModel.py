import torch
import torch.nn as nn
import torch.nn.functional as F

class SurnameGenerationModel(nn.Module):
    def __init(self, char_embedding_size, char_vocab_size, rnn_hidden_size, batch_first = True, padding_idx =0, dropout_p = 0.5):
        super().__init__()

        self.char_emb = nn.Embedding(num_embeddings=char_vocab_size, embedding_dim = char_embedding_size, padding_idx=padding_idx)
        self.rnn = nn.GRU(input_size= char_embedding_size, hidden_size = rnn_hidden_size,batch_first = batch_first)
        self.fc = nn.Linear(in_features=rnn_hidden_size, out_features=char_vocab_size)
        self.dropout_p = dropout_p

        def forward(self, x_in, apply_softmax=True):
            x_embedded = self.char_emb(x_in)
            y_out , _ = self.rnn(x_embedded)
            # with nationality
            # nationality_embeds = self.nation_embed(nationality_index).unsqueeze(0)
            # y_out, _ = self.rnn(x_embedded, nationality_embedded)

            batch_size, seq_size, feat_size = y_out.shape()

            y_out = y_out.contiguou().view(batch_size*seq_size, feat_size)

            y_out = self.fc(F.dropout(y_out, p=self.dropout_p))
            if apply_softmax:
                y_out = F.softmax(y_out, dim=1)

            new_feat_size = y_out.shape[-1]
            y_out = y_out.view(batch_size, seq_size, new_feat_size)
            return y_out