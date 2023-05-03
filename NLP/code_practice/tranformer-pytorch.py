import torch.nn as nn
import torch
import torch.nn.functional as F
import math, copy, re
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import torchtext
import matplotlib.pyplot as plt
warnings.simplefiler("ignore")
print(torch.__version__)

#create word embedding class 
class Embedding(nn.Module):     #nn.Module - pytorch 모듈 클래스 상속, Neural Network Class 들을 이용하기 전에는 반드시 nn.Module을 상속
    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size : size of vocabulary
            embed_dim : dimension of embeddings
        """

        super(Embedding, self).__init__()   #same as super().__init__(). Embedding class 이름과 self 를 이용하여 현재 클래스 어떤 클래스인지 명시
        self.embed = nn.Embedding(vocab_size, embed_dim)    #torch.nn.Embedding 모듈은 학습 데이터로부터 임베딩 벡터를 생성하는 역할
    
    def forward(self, x):       #nn.Module 을 상속 받고나면, __init__과 forward 메서드는 overide 해줘야 함.
        """
        Args:
            x : input vector
        returns:
            out: embedding vector
        """

        out = self.embed(x)
        return out
    
#generate positional encoding -> generates matrix similar to embedding matrix
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        """
        Args:
            seq_len: length of input sentence
            embed_model_dim : dimension of embedding
        """

        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos/(10000**((2*i)/self.embed_dim)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*(i+1))/self.embed_dim)))
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        """
        Args:
            x: input vector
        Returns :
            X : output
        """

        x = x*math.sqrt(self.embed_dim)
        seq_len = x.size(1)
        x = x+torch.autograd.Variable(self.pe[:,:,seq_len], requires_grad=False) #pe[:,:,seq_len] 모든 행, 모든 열의 seq_len index만
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        """
        Args:
            embed_dim: dimension of embedding vector output
            n_heads : number of self attention heads
        """

        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim  #512
        self.n_heads = n_heads  #8
        self.single_head_dim = int(self.embed_dim/self.n_heads) #512/8 = 64

        #key, query, value matrixes 64x64
        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)   
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim, self.embed_dim)

    def forward(self, key, query, value, mask=None): #batch_size x sequence_length x embedding_dim = 32, 10, 512
        """
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder
        
        Returns:
           output vector from multihead attention
        """
        batch_size = key.size(0)
        seq_length = key.size(1)

        # query dimension can change in decoder during inference.
        # so we can't take general seq_legnth
        seq_length_query = query.size(1)

        # 32x10x512
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim) # 32x10x8x64
        query = query.view(batch_size, seq_length, self.n_heads, self.single_head_dim) # 32x10x8x64
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim) # 32x10x8x64

        k = self.key_matrix(key)
        q = self.key_matrix(query)
        v = self.key_matrix(value)

        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1,2)  #(bsize, n_heads, single_head_dim, seq_len) 
                                        #swap idx -1, 2 position tensor. same as k.transpose(2,-1) or k.transpose(2,3) /(3,2)
        product = torch.mathmul(q,k_adjusted)   # (32x8x10x64) x (32x8x64x10) = (32x8x10x10)

        #fill those positions of product matrix as (-1e20) where mask position are 0
        if mask is not None:
            product = product.masked_fill(mask == 0 , float("-1e20"))

        #dviding by square root of key dimension
        product = product / math.sqrt(self.single_head_dim)     #sqrt(64) = 8

        #applying softmax
        scores = F.softmax(product, dim=-1)

        #multiply with value matrix
        scores= torch.matmul(scores,v)  #(32x8x10x10) x (32x8x10x64) = (32x8x10x64)

        #concatenated output 
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim *self.n_heads)
        # (32x8x10x64) -> (32x10x8x64) -> (32,10,512)

        output = self.out(concat)

        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_fator=4, n_heads=8):
        super(TransformerBlock, self).__init__()
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator which determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(nn.Linear(embed_dim, expansion_factor*embed_dim),
                                          nn.ReLU(),
                                          nn.Linear(expansion_fator*embed_dim, embed_dim)
                                          )
        self.dropout1 = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.2)

    def forward(self, key, query, value):

        attention_out = self.attention(key,query,value) #32x10x512
        attention_residual_out = attention_out + value #32x10x512
        norm1_out = self.dropout1(self.norm(attention_residual_out)) #32x10x512

        feed_fwd_out = self.feed_forward(norm1_out) #32x10x512 -> 21x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out #32x10x512
        norm2_out = self.dropout2(self.norm(feed_fwd_residual_out)) #32x10x512

        return norm2_out
    
class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention
        
    Returns:
        out: output of the encoder
    """
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = Embedding(vocab_size, embed_dim)
        self.positional_encoder = PositionalEncoding(seq_len, embed_dim)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])

    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out,out,out)
        
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(DecoderBlock, self).__init__()
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        self.attention = MultiHeadAttention(embed_dim, n_heads =8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)

    def forward(self, key, query, x, mask):
        attention = self.attention(x,x,x,mask=mask) #32x10x512
        value = self.dropout(self.norm(attention+x))

        out = self.transformer_block(key,query, value)

        return out

class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerDecoder, self).__init__()
        
        # expansion_factor : factor which determines number of linear layers in feed forward layer
        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEncoding(seq_len, embed_dim)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, expansion_factor=4, n_heads=8)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, enc_out, mask):
        """
        Args:
            x: input vector from target
            enc_out : output from encoder layer
            trg_mask: mask for decoder self attention
        Returns:
            out: output vector
        """
        x = self.word_embedding(x)
        x = self.position_embedding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask)

        out = F.softmax(self.fc_out(x))
        return out
    
class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, seq_length, num_layers=2, expansion_factor=4,n_heads=8):
        super(Transformer, self).__init__()

        """  
        Args:
           embed_dim:  dimension of embedding 
           src_vocab_size: vocabulary size of source
           target_vocab_size: vocabulary size of target
           seq_length : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention
        
        """

        self.target_vocab_size = target_vocab_size
        self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor = expansion_factor, n_heads = n_heads)

        self.decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)

    def make_trg_mask(self,src, trg):
        """
        for inference
        Args:
            src: input to encoder 
            trg: input to decoder
        out:
            out_labels : returns final prediction of sequence
        """
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
        out_labels = []
        batch_size, seq_len = src.shape[0], src.shape[1]

        out = trg
        for i in range(seq_len):
            out = self.decoder(out, enc_out, trg_mask)

            #taking the last token
            out = out[:,-1,:]

            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out, axis=0)

        return out_labels
    
    def forward(self, src, trg):
        """
        Args:
            src: input to encoder 
            trg: input to decoder
        out:
            out: final vector which returns probabilities of each target word
        """
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
        
        outputs = self.decoder(trg, enc_out, trg_mask)
        return outputs