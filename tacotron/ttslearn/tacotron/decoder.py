# Acknowledgement: some of the code was adapted from ESPnet
#  Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from math import e
import torch
import torch.nn.functional as F
from torch import float32, nn
from ttslearn.tacotron.attention import LocationSensitiveAttention
from ttslearn.tacotron.attention2 import LocationSensitiveAttention2
from ttslearn.tacotron.attention3 import LocationSensitiveAttention3
from ttslearn.util import make_pad_mask


def decoder_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain("tanh"))

class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.alpha = nn.Linear(input_dim, output_dim)

    def forward(self, energy1, energy2, mask=None):
        weights = self.alpha(energy1) + (1 - self.alpha(energy2))
        weights = weights.squeeze(-1)


        if mask is not None:
            weights.masked_fill_(mask, -float("inf"))

        attention_weights = F.softmax(weights, dim=1)
        
        return attention_weights


class ZoneOutCell(nn.Module):
    def __init__(self, cell, zoneout=0.1):
        super().__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout = zoneout

    def forward(self, inputs, hidden):
        next_hidden = self.cell(inputs, hidden)
        next_hidden = self._zoneout(hidden, next_hidden, self.zoneout)
        return next_hidden

    def _zoneout(self, h, next_h, prob):
        h_0, c_0 = h
        h_1, c_1 = next_h
        h_1 = self._apply_zoneout(h_0, h_1, prob)
        c_1 = self._apply_zoneout(c_0, c_1, prob)
        return h_1, c_1

    def _apply_zoneout(self, h, next_h, prob):
        if self.training:
            mask = h.new(*h.size()).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h


class Prenet(nn.Module):
    """Pre-Net of Tacotron/Tacotron 2.

    Args:
        in_dim (int) : dimension of input
        layers (int) : number of pre-net layers
        hidden_dim (int) : dimension of hidden layer
        dropout (float) : dropout rate
    """

    def __init__(self, in_dim, layers=2, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        prenet = nn.ModuleList()
        for layer in range(layers):
            prenet += [
                nn.Linear(in_dim if layer == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
            ]
        self.prenet = nn.Sequential(*prenet)

    def forward(self, x):
        """Forward step

        Args:
            x (torch.Tensor) : input tensor

        Returns:
            torch.Tensor : output tensor
        """
        for layer in self.prenet:
            # 学習時、推論時の両方で Dropout を適用します」
            x = F.dropout(layer(x), self.dropout, training=True)
        return x


class Decoder(nn.Module):
    """Decoder of Tacotron 2.

    Args:
        encoder_hidden_dim (int) : dimension of encoder hidden layer
        out_dim (int) : dimension of output
        layers (int) : number of LSTM layers
        hidden_dim (int) : dimension of hidden layer
        prenet_layers (int) : number of pre-net layers
        prenet_hidden_dim (int) : dimension of pre-net hidden layer
        prenet_dropout (float) : dropout rate of pre-net
        zoneout (float) : zoneout rate
        reduction_factor (int) : reduction factor
        attention_hidden_dim (int) : dimension of attention hidden layer
        attention_conv_channel (int) : number of attention convolution channels
        attention_conv_kernel_size (int) : kernel size of attention convolution
    """

    def __init__(
        self,
        #encoder_hidden_dim=512,  # エンコーダの隠れ層の次元数
        encoder_hidden_dim=128,  # エンコーダの隠れ層の次元数変更        
        out_dim=80,  # 出力の次元数
        layers=2,  # LSTM 層の数
        hidden_dim=1024,  # LSTM層の次元数
        prenet_layers=2,  # Pre-Net の層の数
        prenet_hidden_dim=256,  # Pre-Net の隠れ層の次元数
        prenet_dropout=0.5,  # Pre-Net の Dropout 率
        zoneout=0.1,  # Zoneout 率
        reduction_factor=1,  # Reduction factor
        attention_hidden_dim=128,  # アテンション層の次元数
        attention_conv_channels=32,  # アテンションの畳み込みのチャネル数
        attention_conv_kernel_size=31,  # アテンションの畳み込みのカーネルサイズ
    ):
        super().__init__()
        self.out_dim = out_dim

        # 注意機構
        self.attention = LocationSensitiveAttention(
            encoder_hidden_dim,
            hidden_dim,
            attention_hidden_dim,
            attention_conv_channels,
            attention_conv_kernel_size,
        )

        # 追加の注意機構
        self.attention2 = LocationSensitiveAttention2(
            encoder_hidden_dim,
            hidden_dim,
            attention_hidden_dim,
            attention_conv_channels,
            attention_conv_kernel_size,
        )

        self.attention3 = LocationSensitiveAttention3(
            encoder_hidden_dim,
            hidden_dim,
            attention_hidden_dim,
            attention_conv_channels,
            attention_conv_kernel_size,
        )
        self.reduction_factor = reduction_factor

        # Pre-Net
        self.prenet = Prenet(out_dim, prenet_layers, prenet_hidden_dim, prenet_dropout)

        # 片方向 LSTM
        self.lstm = nn.ModuleList()
        for layer in range(layers):
            lstm = nn.LSTMCell(
                encoder_hidden_dim*3 + prenet_hidden_dim if layer == 0 else hidden_dim,
                #encoder_hidden_dim*2 + prenet_hidden_dim if layer == 0 else hidden_dim,
                #encoder_hidden_dim + prenet_hidden_dim if layer == 0 else hidden_dim
                hidden_dim,
            )
            self.lstm += [ZoneOutCell(lstm, zoneout)]

        # 出力への projection 層
        proj_in_dim = encoder_hidden_dim*3 + hidden_dim 
        #proj_in_dim = encoder_hidden_dim*2 + hidden_dim 
        #proj_in_dim = encoder_hidden_dim + hidden_dim 
        self.feat_out = nn.Linear(proj_in_dim, out_dim * reduction_factor, bias=False)
        self.prob_out = nn.Linear(proj_in_dim, reduction_factor)
        #self.feat_out = nn.Linear(proj_in_dim, out_dim * reduction_factor, bias=False).to(dtype = torch.float64)
        #self.prob_out = nn.Linear(proj_in_dim, reduction_factor).to(dtype = torch.float64)

        #追加
        self.mymodel = MyModel(128,1)
        self.apply(decoder_init)

    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs

    def forward(self, encoder_outs, in_lens, energy_outs, energy_lens, pitch_outs, pitch_lens, max_lens, decoder_targets=None,):
        #self.apply(decoder_init)
        """Forward step

        Args:
            encoder_outs (torch.Tensor) : encoder outputs
            in_lens (torch.Tensor) : input lengths
            decoder_targets (torch.Tensor) : decoder targets for teacher-forcing.

        Returns:
            tuple: tuple of outputs, stop token prediction, and attention weights
        """


        is_inference = decoder_targets is None

        # Reduction factor に基づくフレーム数の調整
        # (B, Lmax, out_dim) ->  (B, Lmax/r, out_dim)
        if self.reduction_factor > 1 and not is_inference:
            decoder_targets = decoder_targets[
                :, self.reduction_factor - 1 :: self.reduction_factor
            ]

        # デコーダの系列長を保持
        # 推論時は、エンコーダの系列長から経験的に上限を定める
        if is_inference:
            max_decoder_time_steps = int(encoder_outs.shape[1] * 10.0)
        else:
            max_decoder_time_steps = decoder_targets.shape[1]

        # ゼロパディングされた部分に対するマスク
        mask = make_pad_mask(max_lens).to(encoder_outs.device)
        emask = make_pad_mask(max_lens).to(energy_outs.device)
        pmask = make_pad_mask(max_lens).to(pitch_outs.device)

        # LSTM の状態をゼロで初期化
        h_list, c_list = [], []
        for _ in range(len(self.lstm)):
            h_list.append(self._zero_state(encoder_outs))
            c_list.append(self._zero_state(encoder_outs))

        # デコーダの最初の入力
        go_frame = encoder_outs.new_zeros(encoder_outs.size(0), self.out_dim)
        prev_out = go_frame

        # 1 つ前の時刻のアテンション重み
        prev_att_w = None
        self.attention.reset()
        self.attention2.reset()
        self.attention3.reset()
        # 追加した1 つ前の時刻のアテンション重み
        prev_att_w2 = None
        #self.attention2.reset()

        # メインループ
        outs, logits, att_ws, att_w2s = [], [], [], []
        t = 0
        while True:
            # コンテキストベクトル、アテンション重みの計算
            att_c, att_w = self.attention(
                encoder_outs, max_lens, energy_outs, max_lens, h_list[0], prev_att_w, prev_att_w2, mask
            )
            
            #print("コンテキストベクトル")
            #print(att_c.shape)
            #print(att_c)
            #print("アテンション重み")
            #print(att_w.shape)
            #print(att_w)
            # 追加したコンテキストベクトル、アテンション重みの計算
            #eatt_c, patt_c, att_w2 = self.attention2(
            #   energy_outs, energy_lens, pitch_outs, pitch_lens, h_list[0], prev_att_w2, emask, pmask
            #)
            #eatt_c, eatt_w = self.attention2(
            #    energy_outs, energy_lens, encoder_outs, energy_lens, h_list[0], prev_att_w2, prev_att_w, emask
            #)
            eerg = self.attention2(
                energy_outs, max_lens, encoder_outs, max_lens, h_list[0], prev_att_w2, prev_att_w, emask
            )
            #patt_c, patt_w = self.attention3(
            #    pitch_outs, pitch_lens, h_list[0], prev_att_w2, pmask
            #)
            perg = self.attention2(
                pitch_outs, max_lens, encoder_outs, max_lens, h_list[0], prev_att_w2, prev_att_w, pmask
            )

            att_w2 = self.mymodel(eerg, perg, emask)

            eatt_c = torch.sum(
              energy_outs * att_w2.unsqueeze(-1), dim=1
            )

            patt_c = torch.sum(
              pitch_outs * att_w2.unsqueeze(-1), dim=1
            )

            #print("追加したエネルギーのコンテキストベクトル")
            #print(eatt_c.shape)
            #print(eatt_c)
            #print("追加したエネルギ-のアテンション重み")
            #print(eatt_w.shape)
            #print(eatt_w)
            #print("追加した周波数のコンテキストベクトル")
            #print(patt_c.shape)
            #print(patt_c)
            #print("追加した周波数のアテンション重み")
            #print(patt_w.shape)
            #print(patt_w)

            #erg = torch.sqrt(eatt_w * patt_w) 
            #print("ergの大きさ")
            #print(erg.shape)
            #print(erg)

            #合計が1になるように正規化
            #attention_weights = erg / erg.sum(dim = 1, keepdim = True)

            #print("最終的なアテンション重み")
            #print(attention_weights.shape)
            #print(attention_weights)


            #eatt_c2 = torch.sum(
            #  energy_outs * attention_weights.unsqueeze(-1), dim=1
            #)           
            #print("最終的なエネルギーのコンテキストベクトル")
            #print(eatt_c2.shape)
            #print(eatt_c2)            

            #patt_c2 = torch.sum(
            #  pitch_outs * attention_weights.unsqueeze(-1), dim=1
            #)           
            #print("最終的な周波数のコンテキストベクトル")
            #print(patt_c2.shape)
            #print(patt_c2)       


            #print("その他の情報")
            #print(ex_feats)
            # Pre-Net
            prenet_out = self.prenet(prev_out)

            # LSTM
            xs = torch.cat([att_c, eatt_c, patt_c, prenet_out], dim=1)
            #xs = torch.cat([att_c, eatt_c, prenet_out], dim=1)
            #xs = torch.cat([att_c, prenet_out], dim=1)
            #print("xsの大きさ")
            #print(xs.shape)
            #print(xs)
            #print("h_list[0]について")
            #print(h_list[0].shape)
            #print(h_list[0])
            h_list[0], c_list[0] = self.lstm[0](xs, (h_list[0], c_list[0]))
            #print("LSTM後のh_list[0]について")
            #print(h_list[0])
            for i in range(1, len(self.lstm)):
                h_list[i], c_list[i] = self.lstm[i](               
                    h_list[i - 1], (h_list[i], c_list[i])
                )
            #print("h_list[-1]について")
            #print(h_list[-1])
             #出力の計算
            hcs = torch.cat([h_list[-1], att_c, eatt_c, patt_c], dim=1)
            #print("hcsの大きさ")
            #print(hcs.shape)
            #print("hcs")
            #print(hcs)
            #hcs = torch.cat([h_list[-1], att_c, eatt_c], dim=1)
            #hcs = torch.cat([h_list[-1], att_c], dim=1)
            #print("hcsの大きさ")
            #print(hcs.shape)
            #print(hcs)
            #print("Type of the added item in 'hcs':", self.feat_out(hcs).view(encoder_outs.size(0), self.out_dim, -1).dtype)
            
            out = self.feat_out(hcs).view(encoder_outs.size(0), self.out_dim, -1)
            #outs.append(self.feat_out(hcs).view(encoder_outs.size(0), self.out_dim, -1))
            #print("outの大きさ")
            #print(out.shape)
            #print("out")
            #print(out)
            outs.append(out)

            logits.append(self.prob_out(hcs))
            att_ws.append(att_w)
            att_w2s.append(att_w2)

            # 次の時刻のデコーダの入力を更新
            if is_inference:
                prev_out = outs[-1][:, :, -1]  # (1, out_dim)
            else:
                # Teacher forcing
                prev_out = decoder_targets[:, t, :]

            # 累積アテンション重み
            prev_att_w = att_w if prev_att_w is None else prev_att_w + att_w

            # 追加累積アテンション重み
            prev_att_w2 = att_w2 if prev_att_w2 is None else prev_att_w2 + att_w2

            t += 1
            # 停止条件のチェック
            if t >= max_decoder_time_steps:
                break
            if is_inference and (torch.sigmoid(logits[-1]) >= 0.5).any():
                break

        # 各時刻の出力を結合
        logits = torch.cat(logits, dim=1)  # (B, Lmax)
        outs = torch.cat(outs, dim=2)  # (B, out_dim, Lmax)
        att_ws = torch.stack(att_ws, dim=1)  # (B, Lmax, Tmax)
        att_w2s = torch.stack(att_w2s, dim=1)  # (B, Lmax, Tmax)
        if self.reduction_factor > 1:
            outs = outs.view(outs.size(0), self.out_dim, -1)  # (B, out_dim, Lmax)
        
        outs = outs.to(torch.float32)
        return outs, logits, att_ws, att_w2s