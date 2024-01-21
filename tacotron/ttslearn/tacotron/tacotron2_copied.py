import torch
from torch import nn
from ttslearn.tacotron.decoder import Decoder
from ttslearn.tacotron.encoder5 import Encoder
from ttslearn.tacotron.postnet import Postnet  #追加
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from math import e
import torch.nn.functional as F
from torch import float32, nn
from ttslearn.tacotron.attention import LocationSensitiveAttention
from ttslearn.util import make_pad_mask




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












def attention1(encoder_outs, src_lens, decoder_state, att_prev, mask, processed_memory1):
  mlp_enc = nn.Linear(512, 128)
  mlp_dec = nn.Linear(1024, 128, bias=False)
  mlp_att = nn.Linear(32, 128, bias=False)
  loc_conv = nn.Conv1d(
    1,
    32,
    31,
    padding=(31 - 1) // 2,
    bias=False,
  )
  w = nn.Linear(128, 1)

  # エンコーダに全結合層を適用した結果を保持
  if processed_memory1 is None:
    processed_memory1 = mlp_enc(encoder_outs)

  # アテンション重みを一様分布で初期化
  if att_prev is None:
    att_prev = 1.0 - make_pad_mask(src_lens).to(
    device=decoder_state.device, dtype=decoder_state.dtype
    )
  att_prev = att_prev / src_lens.unsqueeze(-1).to(encoder_outs.device)

  # (B, T_enc) -> (B, 1, T_enc) -> (B, conv_channels, T_enc) ->
  # (B, T_enc, conv_channels)
  att_conv = loc_conv(att_prev.unsqueeze(1)).transpose(1, 2)
  # (B, T_enc, hidden_dim)
  att_conv = mlp_att(att_conv)

  # (B, 1, hidden_dim)
  decoder_state = mlp_dec(decoder_state).unsqueeze(1)

  # NOTE: アテンションエネルギーは、デコーダの隠れ状態を入力として、次の2 つに依存します
  # 1) デコーダの前の時刻におけるアテンション重み
  # 2) エンコーダの隠れ状態
  #print("att_convとprocessed_memoryとdecoder_stateの大きさ")
  #print(att_conv.shape)
  #print(self.processed_memory.shape)
  #print(decoder_state.shape)
  erg = w(
    torch.tanh(att_conv + processed_memory1 + decoder_state)
  ).squeeze(-1)

  if mask is not None:
    erg.masked_fill_(mask, -float("inf"))

  attention_weights = F.softmax(erg, dim=1)

  # エンコーダ出力の長さ方向に対して重み付き和を取ります
  attention_context = torch.sum(
      encoder_outs * attention_weights.unsqueeze(-1), dim=1
      )

  return attention_context, attention_weights, processed_memory1


def attention2(encoder_outs, src_lens, decoder_state, att_prev, mask, processed_memory2):
  mlp_enc = nn.Linear(512, 128)
  mlp_dec = nn.Linear(1024, 128, bias=False)
  mlp_att = nn.Linear(32, 128, bias=False)
  loc_conv = nn.Conv1d(
    1,
    32,
    31,
    padding=(31 - 1) // 2,
    bias=False,
  )
  w = nn.Linear(128, 1)

  # エンコーダに全結合層を適用した結果を保持
  if processed_memory2 is None:
    processed_memory2 = mlp_enc(encoder_outs)

  # アテンション重みを一様分布で初期化
  if att_prev is None:
    att_prev = 1.0 - make_pad_mask(src_lens).to(
    device=decoder_state.device, dtype=decoder_state.dtype
    )
  att_prev = att_prev / src_lens.unsqueeze(-1).to(encoder_outs.device)

  # (B, T_enc) -> (B, 1, T_enc) -> (B, conv_channels, T_enc) ->
  # (B, T_enc, conv_channels)
  att_conv = loc_conv(att_prev.unsqueeze(1)).transpose(1, 2)
  # (B, T_enc, hidden_dim)
  att_conv = mlp_att(att_conv)

  # (B, 1, hidden_dim)
  decoder_state = mlp_dec(decoder_state).unsqueeze(1)

  # NOTE: アテンションエネルギーは、デコーダの隠れ状態を入力として、次の2 つに依存します
  # 1) デコーダの前の時刻におけるアテンション重み
  # 2) エンコーダの隠れ状態
  #print("att_convとprocessed_memoryとdecoder_stateの大きさ")
  #print(att_conv.shape)
  #print(self.processed_memory.shape)
  #print(decoder_state.shape)
  erg = w(
    torch.tanh(att_conv + processed_memory2 + decoder_state)
  ).squeeze(-1)

  if mask is not None:
    erg.masked_fill_(mask, -float("inf"))

  attention_weights = F.softmax(erg, dim=1)

  # エンコーダ出力の長さ方向に対して重み付き和を取ります
  attention_context = torch.sum(
      encoder_outs * attention_weights.unsqueeze(-1), dim=1
      )

  return attention_context, attention_weights, processed_memory2


def attention1(encoder_outs, src_lens, decoder_state, att_prev, mask, processed_memory3):
  mlp_enc = nn.Linear(512, 128)
  mlp_dec = nn.Linear(1024, 128, bias=False)
  mlp_att = nn.Linear(32, 128, bias=False)
  loc_conv = nn.Conv1d(
    1,
    32,
    31,
    padding=(31 - 1) // 2,
    bias=False,
  )
  w = nn.Linear(128, 1)

  # エンコーダに全結合層を適用した結果を保持
  if processed_memory3 is None:
    processed_memory3 = mlp_enc(encoder_outs)

  # アテンション重みを一様分布で初期化
  if att_prev is None:
    att_prev = 1.0 - make_pad_mask(src_lens).to(
    device=decoder_state.device, dtype=decoder_state.dtype
    )
  att_prev = att_prev / src_lens.unsqueeze(-1).to(encoder_outs.device)

  # (B, T_enc) -> (B, 1, T_enc) -> (B, conv_channels, T_enc) ->
  # (B, T_enc, conv_channels)
  att_conv = loc_conv(att_prev.unsqueeze(1)).transpose(1, 2)
  # (B, T_enc, hidden_dim)
  att_conv = mlp_att(att_conv)

  # (B, 1, hidden_dim)
  decoder_state = mlp_dec(decoder_state).unsqueeze(1)

  # NOTE: アテンションエネルギーは、デコーダの隠れ状態を入力として、次の2 つに依存します
  # 1) デコーダの前の時刻におけるアテンション重み
  # 2) エンコーダの隠れ状態
  #print("att_convとprocessed_memoryとdecoder_stateの大きさ")
  #print(att_conv.shape)
  #print(self.processed_memory.shape)
  #print(decoder_state.shape)
  erg = w(
    torch.tanh(att_conv + processed_memory3 + decoder_state)
  ).squeeze(-1)

  if mask is not None:
    erg.masked_fill_(mask, -float("inf"))

  attention_weights = F.softmax(erg, dim=1)

  # エンコーダ出力の長さ方向に対して重み付き和を取ります
  attention_context = torch.sum(
      encoder_outs * attention_weights.unsqueeze(-1), dim=1
      )

  return attention_context, attention_weights, processed_memory3

















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
        num_vocab=57,
        num_ex=49,
        embed_dim=512,
        encoder_hidden_dim=512,
        encoder_conv_layers=3,
        encoder_conv_channels=512,
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



    def forward(self, seq, in_lens, decoder_targets, energy_feats, energy_lens, pitch_feats, pitch_lens, ex_feats, ex_lens):
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
        embed_layer = nn.Embedding(56, 512, padding_idx=0)


        convs = nn.ModuleList()
        for layer in range(3):
            in_channels = 512 if layer == 0 else 512
            convs += [
                nn.Conv1d(
                    in_channels,
                    512,
                    5,
                    padding=(5 - 1) // 2,
                    bias=False,  # この bias は不要です
                ),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
            ]



        convs = nn.Sequential(*convs)

        # Bi-LSTM による長期依存関係のモデル化
        blstm = nn.LSTM(
            512, 512 // 2, 1, batch_first=True, bidirectional=True
        )


        seq = embed_layer(seq)
        print("入力の文字埋め込み")
        print(seq)
        seq = convs(seq.transpose(1, 2)).transpose(1, 2)
        print("入力のconvs")
        print(seq)
        out = pack_padded_sequence(seq, in_lens.to("cpu"), batch_first=True, enforce_sorted=False)
        out, _ = blstm(out)
        encoder_outs, _ = pad_packed_sequence(out, batch_first=True)
        print("入力の出力")
        print(encoder_outs)

        #encoder_outs = self.encoder(out, in_lens)

        # 追加エンコーダによる音声のエネルギーと基本周波数の潜在表現の獲得
        #energy_outs, pitch_outs = self.encoder2(energy_feats, energy_lens, pitch_feats, pitch_lens)       
        seq = embed_layer(energy_feats)
        print("エネルギーの文字埋め込み")
        print(seq)        
        seq = convs(seq.transpose(1, 2)).transpose(1, 2)
        print("エネルギーのconvs")
        print(seq)        
        out = pack_padded_sequence(seq, energy_lens.to("cpu"), batch_first=True, enforce_sorted=False)
        out, _ = blstm(out)
        energy_outs, _ = pad_packed_sequence(out, batch_first=True)
        print("エネルギーの出力")
        print(energy_outs)        
        #energy_outs = self.encoder(out, energy_lens)



        seq = embed_layer(pitch_feats)
        print("周波数の文字埋め込み")
        print(seq)   
        seq = convs(seq.transpose(1, 2)).transpose(1, 2)
        print("周波数のconvs")
        print(seq)  
        out = pack_padded_sequence(seq, pitch_lens.to("cpu"), batch_first=True, enforce_sorted=False)
        out, _ = blstm(out)
        pitch_outs, _ = pad_packed_sequence(out, batch_first=True)
        print("周波数の出力")
        print(pitch_outs)  













        is_inference = decoder_targets is None

        # Reduction factor に基づくフレーム数の調整
        # (B, Lmax, out_dim) ->  (B, Lmax/r, out_dim)
        if not is_inference:
            decoder_targets = decoder_targets[
                :, 2 - 1 :: 2
            ]


        # デコーダの系列長を保持
        # 推論時は、エンコーダの系列長から経験的に上限を定める
        if is_inference:
            max_decoder_time_steps = int(encoder_outs.shape[1] * 10.0)
        else:
            max_decoder_time_steps = decoder_targets.shape[1]

        # ゼロパディングされた部分に対するマスク
        mask = make_pad_mask(in_lens).to(encoder_outs.device)
        emask = make_pad_mask(energy_lens).to(energy_outs.device)
        pmask = make_pad_mask(pitch_lens).to(pitch_outs.device)


        lstm = nn.ModuleList()
        for layer in range(2):
          lstm = nn.LSTMCell(
          512*3 + 256 if layer == 0 else 1024,
          1024,
          )
          next_hidden = lstm(lstm, 0.1)
          next_hidden = self._zoneout(hidden, next_hidden, self.zoneout)
          lstm += [ZoneOutCell(lstm, 0.1)]



      
        # LSTM の状態をゼロで初期化
        h_list, c_list = [], []
        for _ in range(len(lstm)):
            #h_list.append(_zero_state(encoder_outs))
            #c_list.append(_zero_state(encoder_outs))
            h_list.append(encoder_outs.new_zeros(encoder_outs.size(0), lstm[0].hidden_size))
            c_list.append(encoder_outs.new_zeros(encoder_outs.size(0), lstm[0].hidden_size))

        # デコーダの最初の入力
        go_frame = encoder_outs.new_zeros(encoder_outs.size(0), 80)
        prev_out = go_frame

        # 1 つ前の時刻のアテンション重み
        prev_att_w = None
        prev_att_w2 = None
        processed_memory1 = None
        processed_memory2 = None
        processed_memory3 = None

        # メインループ
        outs, logits, att_ws, att_w2s = [], [], [], []
        t = 0

        while True:
            # コンテキストベクトル、アテンション重みの計算
            att_c, att_w, processed_memory1 = attention1(
                encoder_outs, in_lens, h_list[0], prev_att_w, mask, processed_memory1
            )
            
            print("コンテキストベクトル")
            print(att_c.shape)
            print(att_c)
            print("アテンション重み")
            print(att_w.shape)
            print(att_w)
            # 追加したコンテキストベクトル、アテンション重みの計算
            #eatt_c, patt_c, att_w2 = self.attention2(
            #   energy_outs, energy_lens, pitch_outs, pitch_lens, h_list[0], prev_att_w2, emask, pmask
            #)
            eatt_c, eatt_w, processed_memory2 = self.attention2(
                energy_outs, energy_lens, h_list[0], prev_att_w2, emask, processed_memory2
            )
            patt_c, patt_w, processed_memory3 = self.attention3(
                pitch_outs, pitch_lens, h_list[0], prev_att_w2, pmask, processed_memory3
            )
            print("追加したエネルギーのコンテキストベクトル")
            print(eatt_c.shape)
            print(eatt_c)
            print("追加したエネルギ-のアテンション重み")
            print(eatt_w.shape)
            print(eatt_w)
            print("追加した周波数のコンテキストベクトル")
            print(patt_c.shape)
            print(patt_c)
            print("追加した周波数のアテンション重み")
            print(patt_w.shape)
            print(patt_w)

            erg = torch.sqrt(eatt_w * patt_w) 
            print("ergの大きさ")
            print(erg.shape)
            print(erg)

            #合計が1になるように正規化
            attention_weights = erg / erg.sum(dim = 1, keepdim = True)

            print("最終的なアテンション重み")
            print(attention_weights.shape)
            print(attention_weights)


            eatt_c2 = torch.sum(
              energy_outs * attention_weights.unsqueeze(-1), dim=1
            )           
            print("最終的なエネルギーのコンテキストベクトル")
            print(eatt_c2.shape)
            print(eatt_c2)            

            patt_c2 = torch.sum(
              pitch_outs * attention_weights.unsqueeze(-1), dim=1
            )           
            print("最終的な周波数のコンテキストベクトル")
            print(patt_c2.shape)
            print(patt_c2)       


            print("その他の情報")
            print(ex_feats)
            # Pre-Net

            prenet = nn.ModuleList()
            for layer in range(2):
              prenet += [
                  nn.Linear(80 if layer == 0 else 256, 256),
                  nn.ReLU(),
              ]
            prenet = nn.Sequential(*prenet)

            for layer in prenet:
              # 学習時、推論時の両方で Dropout を適用します」
              x = F.dropout(layer(prev_out), 0.5, training=True)

            prenet_out = x

            # LSTM
            xs = torch.cat([att_c, eatt_c2, patt_c2, prenet_out], dim=1)
            print("xsの大きさ")
            print(xs.shape)
            print(xs)
            print("h_list[0]について")
            print(h_list[0].shape)
            print(h_list[0])
            lstm = nn.ModuleList()
            for layer in range(2):
              lstm = nn.LSTMCell(
              512*3 + 256 if layer == 0 else 1024,
              1024,
              )
            lstm += [ZoneOutCell(lstm, 0.1)]

            h_list[0], c_list[0] = lstm[0](xs, (h_list[0], c_list[0]))
            print("LSTM後のh_list[0]について")
            print(h_list[0])
            for i in range(1, len(lstm)):
                h_list[i], c_list[i] = lstm[i](
                    h_list[i - 1], (h_list[i], c_list[i])
                )
            print("h_list[-1]について")
            print(h_list[-1])
             #出力の計算
            hcs = torch.cat([h_list[-1], att_c, eatt_c, patt_c, ex_feats], dim=1)
            print("hcsの大きさ")
            print(hcs.shape)
            print(hcs)

            #ここから
            proj_in_dim = 512*3 + 1024 + 4
            feat_out = nn.Linear(proj_in_dim, 80 * 2, bias=False).to(dtype = torch.float64)
            prob_out = nn.Linear(proj_in_dim, 2).to(dtype = torch.float64)
            outs.append(feat_out(hcs).view(encoder_outs.size(0), 80, -1))
            logits.append(prob_out(hcs))
            att_ws.append(att_w)
            att_w2s.append(attention_weights)

            # 次の時刻のデコーダの入力を更新
            if is_inference:
                prev_out = outs[-1][:, :, -1]  # (1, out_dim)
            else:
                # Teacher forcing
                prev_out = decoder_targets[:, t, :]

            # 累積アテンション重み
            prev_att_w = att_w if prev_att_w is None else prev_att_w + att_w

            # 追加累積アテンション重み
            prev_att_w2 = attention_weights if prev_att_w2 is None else prev_att_w2 + attention_weights

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
            outs = outs.view(outs.size(0), 80, -1)  # (B, out_dim, Lmax)
        
        outs = outs.to(torch.float32)
        #return outs, logits, att_ws, att_w2s















        

        #pitch_outs = self.encoder(pitch_feats, pitch_lens)
        # デコーダによるメルスペクトログラム、stop token の予測
        #outs, logits, att_ws, att_w2s = self.decoder(encoder_outs, in_lens, energy_outs, energy_lens, pitch_outs, pitch_lens, ex_feats, decoder_targets)


        # Post-Net によるメルスペクトログラムの残差の予測
        #outs_fine = outs + self.postnet(outs)

        postnet = nn.ModuleList()
        for layer in range(5):
            in_channels = 80 if layer == 0 else 512
            out_channels = 5 if layer == 5 - 1 else 512
            postnet += [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    5,
                    stride=1,
                    padding=(5 - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            ]
            if layer != 5 - 1:
                postnet += [nn.Tanh()]
            postnet += [nn.Dropout(0.5)]
        postnet = nn.Sequential(*postnet)

        # Post-Net によるメルスペクトログラムの残差の予測
        outs_fine = outs + postnet(outs)

















        # (B, C, T) -> (B, T, C)
        outs = outs.transpose(2, 1)
        outs_fine = outs_fine.transpose(2, 1)

        return outs, outs_fine, logits, att_ws, att_w2s

    def inference(self, seq, energy_feats, energy_lens, pitch_feats, pitch_lens, ex_feats, ex_lens):
        """Inference step

        Args:
            seq (torch.Tensor): input sequence

        Returns:
            tuple: tuple of outputs, outputs (after post-net), stop token prediction
                and attention weights.
        """
        seq = seq.unsqueeze(0) if len(seq.shape) == 1 else seq
        in_lens = torch.tensor([seq.shape[-1]], dtype=torch.long, device=seq.device)

        outs, outs_fine, logits, att_ws, att_w2s = self.forward(seq, in_lens, None, energy_feats, energy_lens, pitch_feats, pitch_lens, ex_feats, ex_lens)

        return outs[0], outs_fine[0], logits[0], att_ws[0], att_w2s[0]
