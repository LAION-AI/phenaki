import torch
import torch.nn as nn


class TemporalSpatialAttention(nn.Module):
    def __init__(self, channels, size, frames):
        super(TemporalSpatialAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.frames = frames
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=channels, dim_feedforward=2048, nhead=4, batch_first=True)
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=4, norm=nn.LayerNorm(channels))
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=4, norm=nn.LayerNorm(channels))

    def forward(self, x):
        """
        x: 1 x 20 x 8 x 16 x 16
        1. reshape & permute x: 1 * 20 x 16 x 8*8 = 20 x 8 x 256 -> 20 x 256 x 8
        2. spatial attention: q k v -> 20 x 256 x 8 -> 20 x 256 x 8 * 20 x 8 x 256 = 1 x 256 x 256 -> 1 x 256 x 256 * 1 x 256 x 8
        = 1 x 256 x 8
        3. reshape & permute x: 1 * 16 * 16 x 20 x 8 = 256 x 20 x 8
        4. temporal attention: q k v -> 256 x 20 x 8 -> 256 x 20 x 8 * 256 x 8 x 20 = 256 x 20 x 20 -> 256 x 20 x 20 * 256 x 20 x 8
        = 256 x 20 x 8
        :return: x: 1 x 20 x 8 x 16 x 16
        """
        bs, tz, _, _, _ = x.shape
        x = x.view(bs * tz, self.channels, self.size * self.size).permute(0, 2, 1)  # bs * tz x hz * wz x cz
        x = self.spatial_transformer(x)
        x = x.view(bs, tz, self.channels, self.size, self.size)
        print(x.shape)
        x = x.view(bs, tz, self.size*self.size, self.channels).permute(0, 2, 1, 3).view(bs*self.size*self.size, tz, self.channels)  # bs x hz * wz x tz * cz
        mask = torch.tril(torch.ones(tz, tz, dtype=torch.float32)) * -10000
        x = self.temporal_transformer(x, mask=mask)
        x = x.view(bs, self.size, self.size, tz, self.channels).permute(0, 3, 4, 1, 2)
        return x


class Encoder(nn.Module):
    def __int__(self, in_frames=100, out_frames=20, compression=8):
        super(Encoder, self).__int__()
        self.patch_emb = nn.Conv3d(in_frames, out_frames, kernel_size=2, stride=compression)
        # self.attention = TemporalSpatialAttention()

    def forward(self, x):
        # x = self.patch_emb(x)
        x = torch.randn(1, 20, 8, 16, 16)
        print(x.shape)
        bs, tz, cz, wz, hz = x.shape
        tsa = TemporalSpatialAttention(cz, wz, tz)
        x = tsa(x)
        print(x.shape)


if __name__ == '__main__':
    e = Encoder()
    e(1)