class lstm_encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(lstm_encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

