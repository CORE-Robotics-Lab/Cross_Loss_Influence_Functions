# Created by Andrew Silva
"""
Adapted from https://github.com/Adoni/word2vec_pytorch

"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class SkipGramModel(nn.Module):
    """Skip gram model of word2vec.
    Attributes:
        emb_size: Embedding size.
        emb_dimention: Embedding dimention, typically from 50 to 500.
        u_embedding: Embedding for center word.
        v_embedding: Embedding for neibor words.
    """

    def __init__(self, vocab_size, embedding_dim, sparse=True):
        """Initialize model parameters.
        Args:
            vocab_size: size of vocabulary.
            embedding_dim: size of each embedding
        """
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # self.u_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, sparse=True)
        # self.v_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, sparse=True)
        self.u_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, sparse=sparse)
        self.v_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, sparse=sparse)
        init_range = 1/np.sqrt(vocab_size + embedding_dim)
        self.u_embeddings.weight.data.uniform_(-init_range, init_range)
        self.v_embeddings.weight.data.uniform_(-0, 0)
        self.log_sig = nn.LogSigmoid()

    def forward(self, targets, contexts, negatives):
        """
        Args:
            targets: target word ids
            contexts: context word ids
            negatives: negative word ids
        Returns:
            negative sampling loss
        """
        emb_u = self.u_embeddings(targets)
        emb_v = self.v_embeddings(contexts)
        target_context = torch.mul(emb_u, emb_v)
        target_context = torch.sum(target_context, dim=1)
        target_context = self.log_sig(target_context)

        neg_emb_v = self.v_embeddings(negatives)  # .permute(1, 0, 2)  # Move batch dimension to the front
        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)

        # return -1 * (torch.sum(target_context)+torch.sum(neg_score))
        return -(torch.mean(target_context) + torch.mean(neg_score))/2

    def forward_no_negatives(self, targets, contexts):
        """
                Args:
                    targets: target word ids
                    contexts: context word ids
                    negatives: negative word ids
                Returns:
                    negative sampling loss
                """
        emb_u = self.u_embeddings(targets)
        emb_v = self.v_embeddings(contexts)
        target_context = torch.mul(emb_u, emb_v)
        target_context = torch.sum(target_context, dim=1)
        target_context = self.log_sig(target_context)

        return -(torch.mean(target_context))

    def predict(self, token):
        return self.u_embeddings(token)

    def predict_diff(self, token1, token2):
        return self.u_embeddings(token1) - self.v_embeddings(token2)




