from torch import nn

class SimplestEncoder(nn.Module):
    def __init__(self, num_vocab=56, embed_dim=256):
        super().__init__()
        self.embed = nn.Embedding(num_vocab, embed_dim, padding_idx=0)

    def forward(self, seqs):
        return self.embed(seqs)