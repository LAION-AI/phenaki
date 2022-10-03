import torch
import torch.nn as nn


class TemporalSpatialAttention(nn.Module):
    def __init__(self, channels, size, frames):
        super(TemporalSpatialAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.frames = frames
        self.temporal_attention = nn.MultiheadAttention(channels*size*size, 4, batch_first=True)
        self.spatial_attention = nn.MultiheadAttention(channels*frames, 4, batch_first=True)
        self.ln_t = nn.LayerNorm([channels*size*size])
        self.ln_s = nn.LayerNorm([channels*frames])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels*frames]),
            nn.Linear(channels*frames, channels*frames),
            nn.GELU(),
            nn.Linear(channels*frames, channels*frames),
        )

    def forward(self, x):
        bs, tz, _, _, _ = x.shape
        x = x.view(bs, tz, self.channels * self.size * self.size)  # bs x tz x cz * hz * wz
        x_ln = self.ln_t(x)
        attention_value, _ = self.temporal_attention(x_ln, x_ln, x_ln)
        x = attention_value + x
        # x = x.view(bs, tz, self.channels, self.size, self.size)
        x = x.view(bs, tz*self.channels, self.size*self.size).permute(0, 2, 1)  # bs x hz * wz x tz * cz
        x = self.ln_s(x)
        attention_value, _ = self.spatial_attention(x, x, x)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.permute(0, 2, 1).view(bs, tz, self.channels, self.size, self.size)


class Encoder(nn.Module):
    def __int__(self, in_frames=100, out_frames=20, compression=8):
        super(Encoder, self).__int__()
        self.patch_emb = nn.Conv3d(in_frames, out_frames, kernel_size=2, stride=compression)
        # self.attention = TemporalSpatialAttention()

    def forward(self, x):
        # x = self.patch_emb(x)
        x = torch.randn(1, 20, 8, 16, 16)
        bs, tz, cz, wz, hz = x.shape
        tsa = TemporalSpatialAttention(cz, wz, tz)
        x = tsa(x)
        print(x.shape)


if __name__ == '__main__':
    e = Encoder()
    e(1)