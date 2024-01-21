# Acknowledgement: some of the code was adapted from ESPnet
#  Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
from torch import nn
from torch.nn import functional as F
from ttslearn.util import make_pad_mask


class BahdanauAttention(nn.Module):
    """Bahdanau-style attention

    This is an attention mechanism originally used in Tacotron.

    Args:
        encoder_dim (int): dimension of encoder outputs
        decoder_dim (int): dimension of decoder outputs
        hidden_dim (int): dimension of hidden state
    """

    def __init__(self, encoder_dim=512, decoder_dim=1024, hidden_dim=128):
        super().__init__()
        self.mlp_enc = nn.Linear(encoder_dim, hidden_dim)
        self.mlp_dec = nn.Linear(decoder_dim, hidden_dim, bias=False)
        self.w = nn.Linear(hidden_dim, 1)

        self.processed_memory = None

    def reset(self):
        """Reset the internal buffer"""
        self.processed_memory = None

    def forward(
        self,
        energy_outs,
        energy_lens,
        decoder_state,
        mask=None,
    ):
        """Forward step

        Args:
            encoder_outs (torch.FloatTensor): encoder outputs
            src_lens (list): length of each input batch
            decoder_state (torch.FloatTensor): decoder hidden state
            mask (torch.FloatTensor): mask for padding
        """
        # エンコーダに全結合層を適用した結果を保持
        if self.processed_memory is None:
            self.processed_memory = self.mlp_enc(encoder_outs)

        # (B, 1, hidden_dim)
        decoder_state = self.mlp_dec(decoder_state).unsqueeze(1)

        # NOTE: アテンションエネルギーは、デコーダの隠れ状態を入力として、
        # エンコーダの特徴量のみによって決まる
        erg = self.w(torch.tanh(self.processed_memory + decoder_state)).squeeze(-1)

        if mask is not None:
            erg.masked_fill_(mask, -float("inf"))

        attention_weights = F.softmax(erg, dim=1)

        # エンコーダ出力の長さ方向に対して重み付き和を取ります
        attention_context = torch.sum(
            encoder_outs * attention_weights.unsqueeze(-1), dim=1
        )

        return attention_context, attention_weights


class LocationSensitiveAttention2(nn.Module):
    """Location-sensitive attention

    This is an attention mechanism used in Tacotron 2.

    Args:
        encoder_dim (int): dimension of encoder outputs
        decoder_dim (int): dimension of decoder outputs
        hidden_dim (int): dimension of hidden state
        conv_channels (int): number of channels of convolutional layer
        conv_kernel_size (int): size of convolutional kernel
    """

    def __init__(
        self,
        #encoder_dim=512,
        encoder_dim=128,
        decoder_dim=1024,
        hidden_dim=128,
        conv_channels=32,
        conv_kernel_size=31,
    ):
        super().__init__()
        self.mlp_enc = nn.Linear(encoder_dim, hidden_dim)
        self.mlp_enc2 = nn.Linear(encoder_dim, hidden_dim)
        self.mlp_dec = nn.Linear(decoder_dim, hidden_dim, bias=False)
        self.mlp_att = nn.Linear(conv_channels, hidden_dim, bias=False)
        self.mlp_att2 = nn.Linear(conv_channels, hidden_dim, bias=False)
        assert conv_kernel_size % 2 == 1
        self.loc_conv = nn.Conv1d(
            1,
            conv_channels,
            conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2,
            bias=False,
        )

        self.loc_conv2 = nn.Conv1d(
            1,
            conv_channels,
            conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2,
            bias=False,
        )

        #self.w = nn.Linear(hidden_dim, 1)
        self.w = nn.Linear(hidden_dim, 128)

        self.processed_memory = None
        self.processed_memory2 = None

    def reset(self):
        """Reset the internal buffer"""
        self.processed_memory = None
        self.processed_memory2 = None

    def forward(
        self,
        energy_outs,
        energy_lens,
        encoder_outs,
        src_lens,
        decoder_state,
        att_prev2,
        att_prev,
        mask=None,
    ):
        """Forward step

        Args:
            encoder_outs (torch.FloatTensor): encoder outputs
            src_lens (list): length of each input batch
            decoder_state (torch.FloatTensor): decoder hidden state
            att_prev (torch.FloatTensor): previous attention weight
            mask (torch.FloatTensor): mask for padding
        """
        # エンコーダに全結合層を適用した結果を保持
        if self.processed_memory is None:
            self.processed_memory = self.mlp_enc(energy_outs)
      
        # エンコーダに全結合層を適用した結果を保持
        if self.processed_memory2 is None:
            self.processed_memory2 = self.mlp_enc2(encoder_outs)

        # アテンション重みを一様分布で初期化
        if att_prev is None:
            att_prev = 1.0 - make_pad_mask(energy_lens).to(
                device=decoder_state.device, dtype=decoder_state.dtype
            )
            att_prev = att_prev / energy_lens.unsqueeze(-1).to(energy_outs.device)


        # アテンション重みを一様分布で初期化
        if att_prev2 is None:
            att_prev2 = 1.0 - make_pad_mask(src_lens).to(
                device=decoder_state.device, dtype=decoder_state.dtype
            )
            att_prev2 = att_prev2 / src_lens.unsqueeze(-1).to(encoder_outs.device)


        # (B, T_enc) -> (B, 1, T_enc) -> (B, conv_channels, T_enc) ->
        # (B, T_enc, conv_channels)
        att_conv = self.loc_conv(att_prev.unsqueeze(1)).transpose(1, 2)
        # (B, T_enc, hidden_dim)
        att_conv = self.mlp_att(att_conv)


        # (B, T_enc) -> (B, 1, T_enc) -> (B, conv_channels, T_enc) ->
        # (B, T_enc, conv_channels)
        att_conv2 = self.loc_conv2(att_prev2.unsqueeze(1)).transpose(1, 2)
        # (B, T_enc, hidden_dim)
        att_conv2 = self.mlp_att2(att_conv2)

        # (B, 1, hidden_dim)
        decoder_state = self.mlp_dec(decoder_state).unsqueeze(1)

        # NOTE: アテンションエネルギーは、デコーダの隠れ状態を入力として、次の2 つに依存します
        # 1) デコーダの前の時刻におけるアテンション重み
        # 2) エンコーダの隠れ状態
        erg = self.w(
            torch.tanh(att_conv + att_conv2 + self.processed_memory + decoder_state)
        )

        #if mask is not None:
        #    erg.masked_fill_(mask, -float("inf"))

        #attention_weights = F.softmax(erg, dim=1)

        # エンコーダ出力の長さ方向に対して重み付き和を取ります
        #attention_context = torch.sum(
        #    encoder_outs * attention_weights.unsqueeze(-1), dim=1
        #)

        #return attention_context, attention_weights
        return erg


# 書籍中の数式に沿って、わかりやすさを重視した実装
class BahdanauAttentionNaive(nn.Module):
    def __init__(self, encoder_dim=512, decoder_dim=1024, hidden_dim=128):
        super().__init__()
        self.V = nn.Linear(encoder_dim, hidden_dim)
        self.W = nn.Linear(decoder_dim, hidden_dim, bias=False)
        # NOTE: 本書の数式通りに実装するなら bias=False ですが、実用上は bias=True としても問題ありません
        self.w = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        encoder_outs,
        decoder_state,
        mask=None,
    ):
        # 式 (9.11) の計算
        erg = self.w(
            torch.tanh(self.W(decoder_state).unsqueeze(1) + self.V(encoder_outs))
        ).squeeze(-1)

        if mask is not None:
            erg.masked_fill_(mask, -float("inf"))

        attention_weights = F.softmax(erg, dim=1)

        # エンコーダ出力の長さ方向に対して重み付き和を取ります
        attention_context = torch.sum(
            encoder_outs * attention_weights.unsqueeze(-1), dim=1
        )

        return attention_context, attention_weights


# 書籍中の数式に沿って、わかりやすさを重視した実装
class LocationSensitiveAttentionNaive(nn.Module):
    def __init__(
        self,
        encoder_dim=512,
        decoder_dim=1024,
        hidden_dim=128,
        conv_channels=32,
        conv_kernel_size=31,
    ):
        super().__init__()
        self.V1 = nn.Linear(encoder_dim, hidden_dim)
        self.W1 = nn.Linear(decoder_dim, hidden_dim, bias=False)
        self.U1 = nn.Linear(conv_channels, hidden_dim, bias=False)
        self.F1 = nn.Conv1d(
            1,
            conv_channels,
            conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2,
            bias=False,
        )
        # NOTE: 本書の数式通りに実装するなら bias=False ですが、実用上は bias=True としても問題ありません
        self.w1 = nn.Linear(hidden_dim, 1)
        self.V2 = nn.Linear(encoder_dim, hidden_dim)
        self.W2 = nn.Linear(decoder_dim, hidden_dim, bias=False)
        self.U2 = nn.Linear(conv_channels, hidden_dim, bias=False)
        self.F2 = nn.Conv1d(
            1,
            conv_channels,
            conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2,
            bias=False,
        )
        # NOTE: 本書の数式通りに実装するなら bias=False ですが、実用上は bias=True としても問題ありません
        self.w2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        energy_outs,
        energy_lens,
        pitch_outs,
        pitch_lens,
        decoder_state,
        att_prev,
        emask=None,
        pmask=None,
    ):
        # energyアテンション重みを一様分布で初期化
        if att_prev is None:
            att_prev = 1.0 - make_pad_mask(energy_lens).to(
                device=decoder_state.device, dtype=decoder_state.dtype
            )
            att_prev = att_prev / energy_lens.unsqueeze(-1).to(energy_outs.device)

        # (B, T_enc) -> (B, 1, T_enc) -> (B, conv_channels, T_enc) ->
        # (B, T_enc, conv_channels)
        f1 = self.F1(att_prev.unsqueeze(1)).transpose(1, 2)

        # 式 (9.13) の計算
        erg1 = self.w1(
            torch.tanh(
                self.W1(decoder_state).unsqueeze(1) + self.V1(energy_outs) + self.U1(f1)
            )
        ).squeeze(-1)

        if emask is not None:
            erg1.masked_fill_(emask, -float("inf"))

        #attention_weights1 = F.softmax(erg, dim=1)



        #pitchに対して
        if att_prev is None:
            att_prev = 1.0 - make_pad_mask(pitch_lens).to(
                device=decoder_state.device, dtype=decoder_state.dtype
            )
            att_prev = att_prev / pitch_lens.unsqueeze(-1).to(pitch_outs.device)

        # (B, T_enc) -> (B, 1, T_enc) -> (B, conv_channels, T_enc) ->
        # (B, T_enc, conv_channels)
        f2 = self.F2(att_prev.unsqueeze(1)).transpose(1, 2)

        # 式 (9.13) の計算
        erg2 = self.w2(
            torch.tanh(
                self.W2(decoder_state).unsqueeze(1) + self.V2(pitch_outs) + self.U2(f2)
            )
        ).squeeze(-1)

        if pmask is not None:
            erg2.masked_fill_(pmask, -float("inf"))

        #attention_weights2 = F.softmax(erg, dim=1)

        #attention_weights = torch.sqrt(attention_weights1 * attention_weights2)
        erg = torch.sqrt(erg1 * erg2)

        attention_weights = F.softmax(erg, dim=1)

        # エンコーダ出力の長さ方向に対して重み付き和を取ります
        energy_attention_context = torch.sum(
            energy_outs * attention_weights.unsqueeze(-1), dim=1
        )

        pitch_attention_context = torch.sum(
            pitch_outs * attention_weights.unsqueeze(-1), dim=1
        )
        return energy_attention_context, pitch_attention_context, attention_weights
