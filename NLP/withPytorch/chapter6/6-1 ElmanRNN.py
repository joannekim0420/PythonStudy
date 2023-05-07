import torch.nn as nn
import torch.nn.functional as F
import torch

class ElmanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first = False):
        super(ElmanRNN, self).__init__()
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)

        self.batch_first = batch_first  #is the first dimension batch
        self.hidden_size = hidden_size

        def _initialize_hidden(self, batch_size):
            return torch.zeros(batch_size, self.hidden_size)

        def forward(self, x_in, initial_hidden= None):
            """
            :param x_in: if self.batch_first: x_in.shape = (batch_size, seq_size, feat_size)
                        else: x_in.shape = (seq_size, batch_size, feat_size)
            :param initial_hidden: RNN의 초기 은닉상태
            :return:
                    hiddens (torch.Tensor) : 각 타임 스텝에서 RNN 출력
                    if self.batch_frst :
                        hiddens.shape = (batch_size, seq_size, hidden_size)
                    else:
                        hiddens.shape = (seq_size, batch_size, hidden_size)
            """
            if self.batch_first:
                batch_size, seq_size, feat_size = x_in.size()
                x_in = x_in.permute(1,0,2)
            else:
                seq_size, batch_size, feat_size = x_in.size()

            hiddens = []

            if initial_hidden is None:
                initial_hidden = self._initialize_hidden(batch_size)
                initial_hidden = initial_hidden.to(x_in.device)

            hidden_t = initial_hidden

            for t in range(seq_size):
                hidden_t = self.rnn_cell(x_in[t], hidden_t)
                hiddens.append(hidden_t)

            hiddens = torch.stack(hiddens)

            if self.batch_first:
                hiddens = hiddens.permute(1,0,2)
            return hiddens
        
