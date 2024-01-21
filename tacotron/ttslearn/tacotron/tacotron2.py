import torch
from torch import nn
from ttslearn.tacotron.decoder import Decoder
from ttslearn.tacotron.encoder import Encoder
from ttslearn.tacotron.postnet import Postnet
from ttslearn.tacotron.encoder3 import Encoder3 #追加
from ttslearn.tacotron.encoder4 import Encoder4 #追加
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Tacotron2(nn.Module):
    """Tacotron 2

    This implementation does not include the WaveNet vocoder of the Tacotron 2.

    Args:
        num_vocab (int): the size of vocabulary
        embed_dim (int): dimension of embedding
        encoder_hidden_dim (int): dimension of hidden unit
        encoder_conv_layers (int): the number of convolution layers
        encoder_conv_channels (int): the number of convolution channels
        encoder_conv_kernel_size (int): kernel size of convolution
        encoder_dropout (float): dropout rate of convolution
        attention_hidden_dim (int): dimension of hidden unit
        attention_conv_channels (int): the number of convolution channels
        attention_conv_kernel_size (int): kernel size of convolution
        decoder_out_dim (int): dimension of output
        decoder_layers (int): the number of decoder layers
        decoder_hidden_dim (int): dimension of hidden unit
        decoder_prenet_layers (int): the number of prenet layers
        decoder_prenet_hidden_dim (int): dimension of hidden unit
        decoder_prenet_dropout (float): dropout rate of prenet
        decoder_zoneout (float): zoneout rate
        postnet_layers (int): the number of postnet layers
        postnet_channels (int): the number of postnet channels
        postnet_kernel_size (int): kernel size of postnet
        postnet_dropout (float): dropout rate of postnet
        reduction_factor (int): reduction factor
    """

    def __init__(
        self,
        num_vocab=56,
        #embed_dim=512,
        #encoder_hidden_dim=512,
        embed_dim=128,
        encoder_hidden_dim=128,
        encoder_conv_layers=3,
        #encoder_conv_channels=512,
        encoder_conv_channels=128,
        encoder_conv_kernel_size=5,
        encoder_dropout=0.5,
        attention_hidden_dim=128,
        attention_conv_channels=32,
        attention_conv_kernel_size=31,
        decoder_out_dim=80,
        decoder_layers=2,
        decoder_hidden_dim=1024,
        decoder_prenet_layers=2,
        decoder_prenet_hidden_dim=256,
        decoder_prenet_dropout=0.5,
        decoder_zoneout=0.1,
        postnet_layers=5,
        postnet_channels=512,
        postnet_kernel_size=5,
        postnet_dropout=0.5,
        reduction_factor=1,
        conv_layers = 3,
        conv_channels = 512,
        conv_kernel_size = 5,
        dropout = 0.5,
        hidden_dim = 512,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_vocab,
            embed_dim,
            encoder_hidden_dim,
            encoder_conv_layers,
            encoder_conv_channels,
            encoder_conv_kernel_size,
            encoder_dropout,
        )
        self.encoder3 = Encoder3( #追加
            num_vocab,
            embed_dim,
            encoder_hidden_dim,
            encoder_conv_layers,
            encoder_conv_channels,
            encoder_conv_kernel_size,
            encoder_dropout,
        )
        self.encoder4 = Encoder4( #追加
            num_vocab,
            embed_dim,
            encoder_hidden_dim,
            encoder_conv_layers,
            encoder_conv_channels,
            encoder_conv_kernel_size,
            encoder_dropout,
        )
        self.decoder = Decoder(
            encoder_hidden_dim,
            decoder_out_dim,
            decoder_layers,
            decoder_hidden_dim,
            decoder_prenet_layers,
            decoder_prenet_hidden_dim,
            decoder_prenet_dropout,
            decoder_zoneout,
            reduction_factor,
            attention_hidden_dim,
            attention_conv_channels,
            attention_conv_kernel_size,
        )
        self.postnet = Postnet(
            decoder_out_dim,
            postnet_layers,
            postnet_channels,
            postnet_kernel_size,
            postnet_dropout,
        )
        # 文字の埋め込み表現
        self.embed = nn.Embedding(num_vocab, embed_dim, padding_idx=0)
        # 1 次元畳み込みの重ね合わせ：局所的な時間依存関係のモデル化
        convs = nn.ModuleList()
        for layer in range(conv_layers):
            in_channels = embed_dim if layer == 0 else conv_channels
            convs += [
                nn.Conv1d(
                    in_channels,
                    conv_channels,
                    conv_kernel_size,
                    padding=(conv_kernel_size - 1) // 2,
                    bias=False,  # この bias は不要です
                ),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.convs = nn.Sequential(*convs)
        # Bi-LSTM による長期依存関係のモデル化
        self.blstm = nn.LSTM(
            conv_channels, hidden_dim // 2, 1, batch_first=True, bidirectional=True
        )




    def forward(self, seq, in_lens, energy_feats, energy_lens, pitch_feats, pitch_lens, max_lens, decoder_targets):
        """Forward step

        Args:
            seq (torch.Tensor): input sequence
            in_lens (torch.Tensor): input sequence lengths
            decoder_targets (torch.Tensor): target sequence

        Returns:
            tuple: tuple of outputs, outputs (after post-net), stop token prediction
                and attention weights.
        """


        # エンコーダによるテキストの潜在表現の獲得
        encoder_outs = self.encoder(seq, max_lens)

        # 追加エンコーダによる音声のエネルギーと基本周波数の潜在表現の獲得
        energy_outs = self.encoder3(energy_feats, max_lens)
        pitch_outs = self.encoder4(pitch_feats, max_lens)



        #print("入力のシークエンス")
        #print(seq[0])
        #emb = self.embed(seq)
        #print("文字埋め込み")
        #print(emb[0])
        # 1 次元畳み込みと embedding では、入力の shape が異なるので注意
        #out = self.convs(emb.transpose(1, 2)).transpose(1, 2)

        #print("1次元畳み込み")        
        #print(out[0])

        # Bi-LSTM の計算
        #out = pack_padded_sequence(out, in_lens.to("cpu"), batch_first=True, enforce_sorted=False)
        #out, _ = self.blstm(out)
        #encoder_outs, _ = pad_packed_sequence(out, batch_first=True)
        #print("Bi-LSTM")
        #print(encoder_outs[0])

        #emb = self.embed(energy_feats)
        #print(emb[0])
        # 1 次元畳み込みと embedding では、入力の shape が異なるので注意
        #out = self.convs(emb.transpose(1, 2)).transpose(1, 2)
        #print(out[0])

        # Bi-LSTM の計算
        #out = pack_padded_sequence(out, energy_lens.to("cpu"), batch_first=True, enforce_sorted=False)
        #out, _ = self.blstm(out)
        #energy_outs, _ = pad_packed_sequence(out, batch_first=True)

        #emb = self.embed(pitch_feats)
        #print(emb[0])
        # 1 次元畳み込みと embedding では、入力の shape が異なるので注意
        #out = self.convs(emb.transpose(1, 2)).transpose(1, 2)
        #print(out[0])

        # Bi-LSTM の計算
        #out = pack_padded_sequence(out, pitch_lens.to("cpu"), batch_first=True, enforce_sorted=False)
        #out, _ = self.blstm(out)
        #pitch_outs, _ = pad_packed_sequence(out, batch_first=True)



        # デコーダによるメルスペクトログラム、stop token の予測
        outs, logits, att_ws, att_w2s = self.decoder(encoder_outs, in_lens, energy_outs, energy_lens, pitch_outs, pitch_lens, max_lens, decoder_targets)

 
        # Post-Net によるメルスペクトログラムの残差の予測
        outs_fine = outs + self.postnet(outs)

        # (B, C, T) -> (B, T, C)
        outs = outs.transpose(2, 1)
        outs_fine = outs_fine.transpose(2, 1)


        return outs, outs_fine, logits, att_ws, att_w2s

    def inference(self, seq, energy_feats, pitch_feats):
        """Inference step

        Args:
            seq (torch.Tensor): input sequence

        Returns:
            tuple: tuple of outputs, outputs (after post-net), stop token prediction
                and attention weights.
        """
        seq = seq.unsqueeze(0) if len(seq.shape) == 1 else seq
        in_lens = torch.tensor([seq.shape[-1]], dtype=torch.long, device=seq.device)
        energy_feats = energy_feats.unsqueeze(0) if len(energy_feats.shape) == 1 else energy_feats
        energy_lens = torch.tensor([energy_feats.shape[-1]], dtype=torch.long, device=energy_feats.device)
        pitch_feats = pitch_feats.unsqueeze(0) if len(pitch_feats.shape) == 1 else pitch_feats
        pitch_lens = torch.tensor([pitch_feats.shape[-1]], dtype=torch.long, device=pitch_feats.device)
        max_lens = torch.max(torch.stack([in_lens, energy_lens, pitch_lens]), dim=0).values

        outs, outs_fine, logits, att_ws, att_w2s = self.forward(seq, in_lens, energy_feats, energy_lens, pitch_feats, pitch_lens, max_lens, None)

        return outs[0], outs_fine[0], logits[0], att_ws[0], att_w2s[0]