import torch.nn as nn


class FNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        token_num,
        seq_len,
        hidden_dim=None,
        embedding_dim=None,
        use_direct_connection=True,
    ):
        super().__init__()
        self.token_num = token_num
        self.seq_len = seq_len
        if embedding_dim is None:
            embedding_dim = 30
        self.embedding = nn.Embedding(token_num, embedding_dim=embedding_dim)
        self.use_direct_connection = use_direct_connection
        if self.use_direct_connection:
            self.direct_connection_fc = nn.Linear(seq_len * embedding_dim, token_num)
        else:
            print("no direct_connection_fc")

        if hidden_dim is None:
            hidden_dim = 100
        self.fc2 = nn.Linear(seq_len * embedding_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, token_num)
        self.activation = nn.LogSoftmax()

    def forward(self, input):
        embedding = self.embedding(input).view(input.shape[0], -1)
        output = self.fc3(self.fc2(embedding).tanh())
        if self.use_direct_connection:
            output += self.direct_connection_fc(embedding)
        return self.activation(output)
