import torch
from torch import nn
import random

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_idx, num_layers = 1, dropout = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim,
            padding_idx = pad_idx
        )

        self.gru = nn.GRU(
            input_size = embedding_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0.0,
            bidirectional = False
        )

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor):
        # src: (batch, src_len)
        embedded = self.embedding(src) # (batch, src_len, emb)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first = True,
            enforce_sorted = False
        )

        outputs, hidden = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first = True)
        return outputs, hidden # outputs: (batch, src_len, hidden_dim), hidden: (num_layers, batch, hidden_dim)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor, src_lengths: torch.Tensor):
        # hidden: (num_layers, batch, hidden_dim) we use last layer
        # encoder_outputs: (batch, src_len, hidden_dim)
        batch_size, src_len, hidden_dim = encoder_outputs.size()

        query = hidden[-1] # (batch, hidden_dim)
        scores = torch.bmm(encoder_outputs, query.unsqueeze(2)).squeeze(2) # (batch, src_len)

        device_scores = scores.device
        index_range = torch.arange(src_len, device = device_scores).unsqueeze(0)
        mask = index_range >= src_lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim = 1) # (batch, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1),encoder_outputs).squeeze(1) # (batch, hidden_dim)

        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_idx, num_layers = 1, dropout = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim,
            padding_idx = pad_idx
        )

        self.attention = Attention(hidden_dim)

        self.gru = nn.GRU(
            input_size = embedding_dim + hidden_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0.0
        )

        self.fc_out = nn.Linear(
            in_features = hidden_dim * 2,
            out_features = vocab_size
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor, src_lengths: torch.Tensor):
        # input_step: (batch, )
        embedded = self.dropout(self.embedding(input_step)) # (batch, emb)
        embedded = embedded.unsqueeze(1) # (batch, 1, emb)

        context, attn_weights = self.attention(hidden, encoder_outputs, src_lengths)
        context = context.unsqueeze(1) # (batch, 1, hidden)

        rnn_input = torch.cat([embedded, context], dim = 2) # (batch, 1, emb + hidden)

        output, hidden = self.gru(rnn_input, hidden)
        output = output.squeeze(1) # (batch, hidden)
        context = context.squeeze(1) # (batch, hidden)

        concat = torch.cat([output, context], dim = 1) # (batch, hidden * 2)
        logits = self.fc_out(concat) # (batch, vocab_size)

        return logits, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, sos_idx, eos_idx, pad_idx, max_target_len, vocab_size):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.max_target_len = max_target_len
        self.vocab_size = vocab_size

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor, tgt: torch.Tensor | None = None, teacher_forcing_ratio = 0.5):
        batch_size = src.size(0)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        device_local = src.device

        if tgt is not None:
            max_len = tgt.size(1)
        else:
            max_len = self.max_target_len

        outputs = torch.zeros(batch_size, max_len, self.vocab_size, device = device_local)

        input_step = torch.full((batch_size,), self.sos_idx, dtype = torch.long, device = device_local)

        for t in range(max_len):
            logits, hidden, _ = self.decoder(input_step, hidden, encoder_outputs, src_lengths)
            outputs[:, t, :] = logits

            if tgt is not None and t + 1 < max_len and random.random() < teacher_forcing_ratio:
                input_step = tgt[:, t + 1]
            else:
                input_step = logits.argmax(dim = 1)

        return outputs