import torch
import torch.nn as nn
import math

from modules import ConvLayer, Forecasting_Model
from transformer import (
    WavenetTSEmbedding, 
    WavenetFAEmbedding, 
    FrequencyAttention,
    MultiHeadAttention, 
    PositionalWiseFeedForward
    )


class FREQ_ATT(nn.Module):
    """ FREQ_ATT model class.

    """

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=3,
        gru_n_layers=1,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        dropout=0.2
    ):
        super(FREQ_ATT, self).__init__()
        self.window_size = window_size
        self.out_dim = out_dim
        # gru_hid_dim = n_features
        self.f_dim = 128
        self.conv = ConvLayer(n_features, kernel_size)
        self.conv0 = nn.Linear(2*n_features, self.f_dim)
        #self.conv1 = nn.Linear(3*n_features, self.f_dim)
        #self.conv2 = nn.Linear(self.f_dim, n_features)

        self.enc_seq_embedding = WavenetTSEmbedding(embedding_dim=n_features, input_channel=n_features)
        self.enc_fea_embedding = WavenetFAEmbedding(embedding_dim=window_size, input_channel=window_size)

        self.encoder_layers = nn.ModuleList([])
        self.use_pos_encoding = False
        layers = 2
        K = 4
        heads = 4
        for ind in range(layers):
            self.encoder_layers.append(nn.ModuleList([
                FrequencyAttention(model_dim=self.f_dim, K = K, num_heads=heads, dropout=dropout),
                #MultiHeadAttention(model_dim=self.f_dim, num_heads=heads, dropout=dropout),
                MultiHeadAttention(model_dim=self.f_dim, num_heads=heads, dropout=dropout),
                PositionalWiseFeedForward(model_dim=self.f_dim, ffn_dim=self.f_dim, dropout=dropout)
            ]))

        self.trend_ff = PositionalWiseFeedForward(model_dim=self.f_dim, ffn_dim=self.f_dim, dropout=dropout)
        self.season_ff = PositionalWiseFeedForward(model_dim=self.f_dim, ffn_dim=self.f_dim, dropout=dropout)

        self.trend_li = nn.Linear(layers*self.f_dim, self.f_dim)
        self.season_li = nn.Linear(layers*self.f_dim, self.f_dim)

        self.decder_layers = nn.Sequential(
            nn.Linear(self.f_dim, self.f_dim),
            nn.ReLU(),
            nn.Linear(self.f_dim, out_dim)
        )

        forecast_hid_dim = self.f_dim
        self.forecasting_model = Forecasting_Model(window_size*self.f_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        #self.recon_model = ReconstructionModel(n_features, recon_hid_dim, out_dim, recon_n_layers, dropout)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # print('x:', x.shape)
        x = self.conv(x)
        #x = self.conv0(x)
        #x = self.enc_seq_embedding(x)
        seq = self.enc_seq_embedding(x)
        fea = self.enc_fea_embedding(x)
        x = torch.cat([seq, fea], dim=2)
        x = self.conv0(x)

        if self.use_pos_encoding:
            pe = torch.ones_like(x[0])
            position = torch.arange(0, self.window_size).unsqueeze(-1)
            temp = torch.Tensor(range(0, x.shape[-1], 2))
            temp = temp * -(math.log(10000) / x.shape[-1])
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)
            x = x + pe

        #x_cat = torch.cat([x, seq, fea], dim=2)
        residual = x #self.conv1(x)
        #print('residual:', residual.shape)
        
        trends = []
        seasonals = []

        #trend_sum = 0
        #season_sum = 0

        for freq_attn, multi_attn, ff_block in self.encoder_layers:
            #seasonal = freq_attn(residual, residual, residual)
            seasonal = freq_attn(residual)
            residual = residual - seasonal
            #print('seasonal:',seasonal.shape)
            trend, _ = multi_attn(residual, residual, residual)
            residual = residual - trend
            #print('trend:',trend.shape)
            #seasonal, _ = multi_attn(seasonal, seasonal, seasonal)
            residual = ff_block(residual)

            trends.append(trend)
            seasonals.append(seasonal)
            #trend_sum += trend 
            #season_sum += seasonal

        #trends = torch.stack(trends, dim=-2)
        #seasonals = torch.stack(seasonals, dim=-2)
        trends = torch.cat(trends, dim=2)
        seasonals = torch.cat(seasonals, dim=2)
        #print(trends.shape)
        #print(seasonals.shape)

        trend_sum = self.trend_li(trends)
        season_sum = self.season_li(seasonals)

        recons = self.trend_ff(trend_sum) + self.season_ff(season_sum) + residual

        dec = self.decder_layers(recons)
        # print('dec:', dec.shape)

        fusion = recons.view(recons.shape[0], -1)
        predictions = self.forecasting_model(fusion)
        # recons = self.recon_model(fusion_f)
        # recons = recons.contiguous().view(recons.shape[0], self.window_size, self.out_dim)
        # print('predictions:', predictions.shape)
        # print('recons:', recons.shape)

        return predictions, dec
