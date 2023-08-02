from transformers import AutoTokenizer, BertModel
from bertviz.transformers_neuron_view import BertModel
from bertviz.neuron_view import show
from transformers import AutoConfig
import torch
from torch import nn
import torch.nn as nn
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


def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1,2))/sqrt(dim_k)
    weights = F.softmax(scores, dim=-1) #torch.Size([1, 5, 5])
    return torch.bmm(weights, value)

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        atten_outputs = scaled_dot_product_attention(self.q(hidden_state),self.k(hidden_state),self.v(hidden_state))
        return atten_outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim = -1)
        # torch.Size([1, 5, 768])
        x = self.output_linear(x)
        return x

multihead_attn = MultiHeadAttention(config)
attn_output = multihead_attn(inputs_embeds)     #torch.Size([1, 5, 768])

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

feed_forward = FeedForward(config)
ff_outputs = feed_forward(attn_output)
print(ff_outputs.size())
#torch.Size([1, 5, 768])

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        hidden_state = self.layer_norm_1(x)
        x = x+self.attention(hidden_state) #prior layer normalization
        x = x+self.feed_forward(self.layer_norm_2(x))
        return x

encoder_layer = TransformerEncoderLayer(config)
print(inputs_embeds.shape, encoder_layer(inputs_embeds).size()) #the shape is the same torch.Size([1, 5, 768])