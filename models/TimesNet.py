import torch
import torch.fft
import torch.nn as nn

from layers.Conv_Blocks import Inception_Block_V1
from layers.Embed import DataEmbedding


def fft_for_period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = fft_for_period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = torch.nn.functional.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, settings):
        super(Model, self).__init__()
        self.settings = settings
        self.seq_len = settings.seq_len
        self.pred_len = settings.pred_len
        self.time_blocks = nn.ModuleList([TimesBlock(settings)
                                          for _ in range(settings.e_layers)])
        self.x_enc_embedding = DataEmbedding(settings.enc_in, settings.d_model, settings.freq, settings.dropout)

        self.y_enc_embedding = DataEmbedding(settings.enc_in - settings.c_out, settings.d_model, settings.dropout,
                                             position_embedding=self.x_enc_embedding.position_embedding,
                                             temporal_embedding=self.x_enc_embedding.temporal_embedding)

        self.layer_norm = nn.LayerNorm(settings.d_model)

        full_len = self.pred_len + self.seq_len
        # self.predict_linear = nn.Linear(self.seq_len, full_len)

        num_concat = 2
        self.concat_block = nn.ModuleList([nn.Linear(full_len, full_len) for _ in range(num_concat)])

        self.projection = nn.Linear(settings.d_model, settings.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, y_enc, y_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True)  # .detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        x_enc_out = self.x_enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        y_enc_out = self.y_enc_embedding(y_enc, y_mark_enc)

        enc_out = torch.concat([x_enc_out, y_enc_out], dim=1)

        enc_out = enc_out.permute(0, 2, 1)
        for linear in self.concat_block:
            enc_out = linear(enc_out)
        enc_out = enc_out.permute(0, 2, 1)

        # x_enc_out = self.predict_linear(x_enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, y_enc, y_mark_enc):
        dec_out = self.forecast(x_enc, x_mark_enc, y_enc, y_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
