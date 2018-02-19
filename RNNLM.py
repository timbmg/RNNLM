import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNLM(nn.Module):

    def __init__(self, vocab_size, **kwargs):

        super().__init__()

        embedding_size  = kwargs.get('embedding_size', 300)
        hidden_size     = kwargs.get('hidden_size', 128)
        num_layers      = kwargs.get('num_layers', 1)
        padding_idx     = kwargs.get('padding_idx', 0)

        self.EMB = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)

        self.RNN = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)

        self.h2w = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, lengths):

        # embedd tokens
        embeddings = self.EMB(inputs)

        # pack embeddings (removes <pad> tokens from sequence)
        packed_embeddings = pack_padded_sequence(embeddings, lengths.data.tolist(), batch_first=True)

        # RNN forward-pass
        packed_outputs, _ = self.RNN(packed_embeddings)

        # get word predictions
        word_logits = self.h2w(packed_outputs.data)

        return word_logits
