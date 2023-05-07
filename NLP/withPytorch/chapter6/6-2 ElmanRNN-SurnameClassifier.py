import torch.nn as nn
import torch.nn.functional as F
import torch
from 6-1 ElmanRNN import ElmanRNN

def column_gather(y_out, x_lengths):
    x_lengths = x_lengths.long().detach().cpu().numpy() -1
    out = []
    for batch_index, column_index in enumerate(x_lengths):
        out.append(y_out[batch_index, column_index])
    return torch.stack(out)


class SurnameClassifier(nn.Module):
    def __init__(self, embedding_size, num_embeddings, num_classes, rnn_hidden_size, batch_first=True, padding_idx=0):
        """
        :param embedding_size: 문자 임베딩의 크기
        :param num_embeddings: 임베딩할 문자 개수
        :param num_classes: 예측 벡터의 크기
        :param padding_idx: 텐서 패딩을 위한 인덱스
        """
        super(SurnameClassifier, self).__init__()

        self.emb = nn.Embedding(num_embeddings = num_embeddings, embedding_dim=embedding_size, padding_idx=padding_idx)
        self.rnn = ElmanRNN(input_size=embedding_size,hidden_size=rnn_hidden_size,batch_first= batch_first)
        self.fc1 = nn.Linear(in_features=rnn_hidden_size, out_features=rnn_hidden_size)
        self.fc2 = nn.Linear(in_features=rnn_hidden_size, out_features=num_classes)

        def forward(self, x_in, x_lengths=None, apply_softmax=False):

            x_embedded = self.emb(x_in)
            y_out = self.rnn(x_embedded)

            if x_lengths is not None:
                y_out = column_gather(y_out, x_lengths)
            else:
                y_out = y_out[:,-1,:]

            y_out = F.dropout(y_out, 0.5)
            y_out = F.relu(self.fc1(y_out))
            y_out = F.dropout(y_out, 0.5)
            y_out = self.fc2(y_out)

            if apply_softmax:
                y_out = F.softmax(y_out, dim=1)
            else:
                y_out
