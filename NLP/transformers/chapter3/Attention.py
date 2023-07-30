from transformers import AutoTokenizer, BertModel
from bertviz.transformers_neuron_view import BertModel
from bertviz.neuron_view import show
from torch import nn
from transformers import AutoConfig
import torch
from math import sqrt
import torch.nn.functional as F

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt)
text= "time flies like an arrow"
# show(model, "bert", tokenizer, text, display_mode="light", layer=0, head=8)

inputs = tokenizer(text, return_tensors ="pt", add_special_tokens =False)
print(inputs.input_ids)
#tensor([[ 2051, 10029,  2066,  2019,  8612]])

config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
print(token_emb)
#Embedding(30522, 768)

inputs_embeds = token_emb(inputs.input_ids)
print(inputs_embeds.size())
#[1,5,768] = [batch_size, seq_len, hidden_dim]


def scaled_dot_product(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1,2))/sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)