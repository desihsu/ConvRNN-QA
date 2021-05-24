import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import cuda, load_cached_embeddings
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _sort_batch_by_length(tensor, sequence_lengths):
    """
    Sorts input sequences by lengths. This is required by Pytorch
    `pack_padded_sequence`. Note: `pack_padded_sequence` has an option to
    sort sequences internally, but we do it by ourselves.

    Args:
        tensor: Input tensor to RNN [batch_size, len, dim].
        sequence_lengths: Lengths of input sequences.

    Returns:
        sorted_tensor: Sorted input tensor ready for RNN [batch_size, len, dim].
        sorted_sequence_lengths: Sorted lengths.
        restoration_indices: Indices to recover the original order.
    """
    # Sort sequence lengths
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    # Sort sequences
    sorted_tensor = tensor.index_select(0, permutation_index)
    # Find indices to recover the original order
    index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths))).long()
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices


class SpanAttention(nn.Module):
    """
    This module returns attention scores over sequence length.

    Args:
        q_dim: Int. Passage vector dimension.

    Inputs:
        q: Question tensor (float), [batch_size, q_len, q_dim].
        q_mask: Question mask (bool), an elements is `False` if it's a word
            `True` if it's a pad token. [batch_size, q_len].

    Returns:
        Attention scores over sequence length, [batch_size, len].
    """
    def __init__(self, q_dim):
        super().__init__()
        self.linear = nn.Linear(q_dim, 1)

    def forward(self, q, q_mask):
        # Compute scores
        q_scores = self.linear(q).squeeze(2)  # [batch_size, len]
        # Assign -inf to pad tokens
        q_scores.data.masked_fill_(q_mask.data, -float('inf'))
        # Normalize along sequence length
        return F.softmax(q_scores, 1)  # [batch_size, len]


class BilinearOutput(nn.Module):
    """
    This module returns logits over the input sequence.

    Args:
        p_dim: Int. Passage hidden dimension.
        q_dim: Int. Question hidden dimension.

    Inputs:
        p: Passage hidden tensor (float), [batch_size, p_len, p_dim].
        q: Question vector tensor (float), [batch_size, q_dim].
        q_mask: Question mask (bool), an elements is `False` if it's a word
            `True` if it's a pad token. [batch_size, q_len].

    Returns:
        Logits over the input sequence, [batch_size, p_len].
    """
    def __init__(self, p_dim, q_dim):
        super().__init__()
        self.linear = nn.Linear(q_dim, p_dim)

    def forward(self, p, q, p_mask):
        # Compute bilinear scores
        q_key = self.linear(q).unsqueeze(2)  # [batch_size, p_dim, 1]
        p_scores = torch.bmm(p, q_key).squeeze(2)  # [batch_size, p_len]
        # Assign -inf to pad tokens
        p_scores.data.masked_fill_(p_mask.data, -float('inf'))
        return p_scores  # [batch_size, p_len]


class EncoderBlock(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.conv1 = ConvBlock(emb_dim, emb_dim, emb_dim)
        self.self_attn = nn.MultiheadAttention(emb_dim, num_heads=6)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = out1.transpose(0,1)
        out2 += self.self_attn(out2, out2, out2)[0]
        out3 = out1 + out2.transpose(0, 1)
        return out3 # [batch_size, seq_len, emb_dim]
 

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.pointwise(self.depthwise(x.transpose(1,2))).transpose(1,2)
        return F.relu(x + out) # [batch_size, seq_len, emb_dim]


class Attention(nn.Module):
    def __init__(self, p_dim):
        super().__init__()
        self.linear = nn.Linear(p_dim, p_dim)
        self.relu = nn.ReLU()

    def forward(self, p, q, q_mask, mask_q=True):
        # Compute scores
        p_key = self.relu(self.linear(p))  # [batch_size, p_len, p_dim]
        q_key = self.relu(self.linear(q))  # [batch_size, q_len, p_dim]
        scores = p_key.bmm(q_key.transpose(2, 1))  # [batch_size, p_len, q_len]
        # Stack question mask p_len times
        if mask_q:
            q_mask = q_mask.unsqueeze(1).repeat(1, scores.size(1), 1)
        else:
            q_mask = q_mask.unsqueeze(2).repeat(1, 1, scores.size(2))
        # Assign -inf to pad tokens
        scores.data.masked_fill_(q_mask.data, -float('inf'))
        # Normalize along question length
        return scores  # [batch_size, p_len, q_len]


class DCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pad_token_id = args.pad_token_id

        # Initialize embedding layer (1)
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)

        # Initialize self attention + convolution encoder (2)
        self.emb_encoder = EncoderBlock(args.embedding_dim)

        # Initialize Context2Query / Query2Context (3)
        self.attn = Attention(args.embedding_dim)

        rnn_cell = nn.LSTM if args.rnn_cell_type == 'lstm' else nn.GRU

        # Initialize passage encoder (4)
        self.passage_rnn = rnn_cell(
            args.embedding_dim * 4,
            args.hidden_dim,
            bidirectional=args.bidirectional,
            batch_first=True,
        )

        # Initialize question encoder (5)
        self.question_rnn = rnn_cell(
            args.embedding_dim,
            args.hidden_dim,
            bidirectional=args.bidirectional,
            batch_first=True,
        )

        self.dropout = nn.Dropout(self.args.dropout)

        # Adjust hidden dimension if bidirectional RNNs are used
        _hidden_dim = (
            args.hidden_dim * 2 if args.bidirectional
            else args.hidden_dim
        )

        # Initialize attention layer for question attentive sum (6)
        self.question_att = SpanAttention(_hidden_dim)

        # Initialize bilinear layer for start positions (7)
        self.start_output = BilinearOutput(_hidden_dim, _hidden_dim)

        # Initialize bilinear layer for end positions (8)
        self.end_output = BilinearOutput(_hidden_dim, _hidden_dim)

    def load_pretrained_embeddings(self, vocabulary, path):
        """
        Loads GloVe vectors and initializes the embedding matrix.

        Args:
            vocabulary: `Vocabulary` object.
            path: Embedding path, e.g. "glove/glove.6B.300d.txt".
        """
        embedding_map = load_cached_embeddings(path)

        # Create embedding matrix. By default, embeddings are randomly
        # initialized from Uniform(-0.1, 0.1).
        embeddings = torch.zeros(
            (len(vocabulary), self.args.embedding_dim)
        ).uniform_(-0.1, 0.1)

        # Initialize pre-trained embeddings.
        num_pretrained = 0
        for (i, word) in enumerate(vocabulary.words):
            if word in embedding_map:
                embeddings[i] = torch.tensor(embedding_map[word])
                num_pretrained += 1

        # Place embedding matrix on GPU.
        self.embedding.weight.data = cuda(self.args, embeddings)

        return num_pretrained

    def sorted_rnn(self, sequences, sequence_lengths, rnn):
        """
        Sorts and packs inputs, then feeds them into RNN.

        Args:
            sequences: Input sequences, [batch_size, len, dim].
            sequence_lengths: Lengths for each sequence, [batch_size].
            rnn: Registered LSTM or GRU.

        Returns:
            All hidden states, [batch_size, len, hid].
        """
        # Sort input sequences
        sorted_inputs, sorted_sequence_lengths, restoration_indices = _sort_batch_by_length(
            sequences, sequence_lengths
        )
        # Pack input sequences
        packed_sequence_input = pack_padded_sequence(
            sorted_inputs,
            sorted_sequence_lengths.data.long().tolist(),
            batch_first=True
        )
        # Run RNN
        packed_sequence_output, _ = rnn(packed_sequence_input, None)
        # Unpack hidden states
        unpacked_sequence_tensor, _ = pad_packed_sequence(
            packed_sequence_output, batch_first=True
        )
        # Restore the original order in the batch and return all hidden states
        return unpacked_sequence_tensor.index_select(0, restoration_indices)

    def forward(self, batch):
        # Obtain masks and lengths for passage and question.
        passage_mask = (batch['passages'] != self.pad_token_id)  # [batch_size, p_len]
        question_mask = (batch['questions'] != self.pad_token_id)  # [batch_size, q_len]
        passage_lengths = passage_mask.long().sum(-1)  # [batch_size]
        question_lengths = question_mask.long().sum(-1)  # [batch_size]

        # 1) Embedding Layer: Embed the passage and question.
        passage_embeddings = self.embedding(batch['passages'])  # [batch_size, p_len, p_dim]
        question_embeddings = self.embedding(batch['questions'])  # [batch_size, q_len, q_dim]

        # 2) Self attention + convolution
        passage_embeddings = self.emb_encoder(passage_embeddings)
        question_embeddings = self.emb_encoder(question_embeddings)

        # 3) Context2Query / Query2Context: Compute weighted sum of question embeddings for
        #       each passage word and concatenate with passage embeddings.
        scores_Q = self.attn(passage_embeddings, question_embeddings, ~question_mask)  # [batch_size, p_len, q_len]
        scores_C = self.attn(passage_embeddings, question_embeddings, ~passage_mask, mask_q=False)

        S = F.softmax(scores_Q, 2) # [batch_size, p_len, q_len]
        A = S.bmm(question_embeddings) # [batch_size, p_len, q_dim]
        S_ = F.softmax(scores_C, 1).transpose(1, 2) # [batch_size, q_len, p_len]
        SS = S.bmm(S_) # [batch_size, p_len, p_len]
        B = SS.bmm(passage_embeddings) # [batch_size, p_len, p_dim]
        C_A = torch.mul(passage_embeddings, A) # [batch_size, p_len, p_dim]
        C_B = torch.mul(passage_embeddings, B) # [batch_size, p_len, p_dim]

        passage_embeddings = cuda(
            self.args,
            torch.cat((passage_embeddings, A, C_A, C_B), 2),
        )  # [batch_size, p_len, p_dim * 4]

        # 4) Passage Encoder
        passage_hidden = self.sorted_rnn(
            passage_embeddings, passage_lengths, self.passage_rnn
        )  # [batch_size, p_len, p_hid]
        passage_hidden = self.dropout(passage_hidden)  # [batch_size, p_len, p_hid]

        # 5) Question Encoder: Encode question embeddings.
        question_hidden = self.sorted_rnn(
            question_embeddings, question_lengths, self.question_rnn
        )  # [batch_size, q_len, q_hid]

        # 6) Question Attentive Sum: Compute weighted sum of question hidden
        #        vectors.
        question_scores = self.question_att(question_hidden, ~question_mask)
        question_vector = question_scores.unsqueeze(1).bmm(question_hidden).squeeze(1)
        question_vector = self.dropout(question_vector)  # [batch_size, q_hid]

        # 7) Start Position Pointer: Compute logits for start positions
        start_logits = self.start_output(
            passage_hidden, question_vector, ~passage_mask
        )  # [batch_size, p_len]

        # 8) End Position Pointer: Compute logits for end positions
        end_logits = self.end_output(
            passage_hidden, question_vector, ~passage_mask
        )  # [batch_size, p_len]

        return start_logits, end_logits  # [batch_size, p_len], [batch_size, p_len]