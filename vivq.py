import torch
import torch.nn as nn
from torchtools.nn import VectorQuantize
from fast_pytorch_kmeans import KMeans


class TemporalSpatialAttention(nn.Module):
    def __init__(self, channels, size, frames, num_layers=4, num_heads=4, spatial_first=True):
        super(TemporalSpatialAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.frames = frames
        self.spatial_first = spatial_first
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=channels, dim_feedforward=2048, nhead=num_heads, batch_first=True)
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=num_layers, norm=nn.LayerNorm(channels))
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=num_layers, norm=nn.LayerNorm(channels))

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
        if self.spatial_first:
            x = x.view(bs * tz, self.channels, self.size * self.size).permute(0, 2, 1)  # bs * tz x hz * wz x cz
            x = self.spatial_transformer(x)
            x = x.view(bs, tz, self.channels, self.size, self.size)
            x = x.view(bs, tz, self.size*self.size, self.channels).permute(0, 2, 1, 3).view(bs*self.size*self.size, tz, self.channels)  # bs x hz * wz x tz * cz
            mask = torch.tril(torch.ones(tz, tz, dtype=torch.float32)) * -10000
            x = self.temporal_transformer(x, mask=mask)
            x = x.view(bs, self.size, self.size, tz, self.channels).permute(0, 3, 4, 1, 2)
        else:
            x = x.view(bs, tz, self.size*self.size, self.channels).permute(0, 2, 1, 3).view(bs*self.size*self.size, tz, self.channels)  # bs x hz * wz x tz * cz
            mask = torch.tril(torch.ones(tz, tz, dtype=torch.float32)) * -10000
            x = self.temporal_transformer(x, mask=mask)
            x = x.view(bs, self.size, self.size, tz, self.channels).permute(0, 3, 4, 1, 2)
            x = x.view(bs * tz, self.channels, self.size * self.size).permute(0, 2, 1)  # bs * tz x hz * wz x cz
            x = self.spatial_transformer(x)
            x = x.view(bs, tz, self.channels, self.size, self.size)
        return x


class Encoder(nn.Module):
    def __init__(self, patch_size=(5, 8, 8), input_channels=3, hidden_channels=64, size=32, compressed_frames=5, num_layers=4, num_heads=4):
        super(Encoder, self).__init__()
        self.patch_emb = nn.Conv3d(input_channels, hidden_channels, kernel_size=patch_size, stride=patch_size)
        self.attention = TemporalSpatialAttention(hidden_channels, size, compressed_frames, num_layers=num_layers, num_heads=num_heads)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.patch_emb(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.attention(x)
        return x


class Decoder(nn.Module):
    def __init__(self, patch_size=(5, 8, 8), input_channels=3, hidden_channels=64, size=32, compressed_frames=5, num_layers=4, num_heads=4):
        super(Decoder, self).__init__()
        self.patch_emb = nn.Conv3d(input_channels, hidden_channels, kernel_size=patch_size, stride=patch_size)
        self.attention = TemporalSpatialAttention(hidden_channels, size, compressed_frames, num_layers=num_layers, num_heads=num_heads, spatial_first=False)

    def forward(self, x):
        x = self.attention(x)

        return x


class VQModule(nn.Module):
    def __init__(self, c_hidden, k, q_init, q_refresh_step, q_refresh_end, reservoir_size=int(9e4)):
        super().__init__()
        self.vquantizer = VectorQuantize(c_hidden, k=k, ema_loss=True)
        self.codebook_size = k
        self.q_init, self.q_refresh_step, self.q_refresh_end = q_init, q_refresh_step, q_refresh_end
        self.register_buffer('q_step_counter', torch.tensor(0))
        self.reservoir = None
        self.reservoir_size = reservoir_size

    def forward(self, x, dim=-1):
        if self.training:
            self.q_step_counter += x.size(0)
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
            self.reservoir = x_flat if self.reservoir is None else torch.cat([self.reservoir, x_flat], dim=0)
            self.reservoir = self.reservoir[torch.randperm(self.reservoir.size(0))[:self.reservoir_size]].detach()
            if self.q_step_counter < self.q_init:
                qe, commit_loss, indices = x, x.new_tensor(0), None
            else:
                if self.q_step_counter < self.q_init + self.q_refresh_end:
                    if (
                            self.q_step_counter + self.q_init) % self.q_refresh_step == 0 or self.q_step_counter == self.q_init or self.q_step_counter == self.q_init + self.q_refresh_end - 1:
                        kmeans = KMeans(n_clusters=self.codebook_size, mode='euclidean', verbose=0)
                        kmeans.fit_predict(self.reservoir)
                        self.vquantizer.codebook.weight.data = kmeans.centroids.detach()
                qe, (_, commit_loss), indices = self.vquantizer(x, dim=dim)
        else:
            if self.q_step_counter < self.q_init:
                qe, commit_loss, indices = x, x.new_tensor(0), None
            else:
                qe, (_, commit_loss), indices = self.vquantizer(x, dim=dim)

        return qe, commit_loss, indices


class VQModel(nn.Module):
    def __init__(self, batch_size=1, compressed_frames=5, latent_size=32, c_hidden=64, c_codebook=16, codebook_size=1024,
                 num_layers_enc=4, num_layers_dec=4, num_heads=4):
        super().__init__()
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.c_hidden = c_hidden
        self.encoder = Encoder(hidden_channels=c_hidden, size=latent_size, compressed_frames=compressed_frames, num_layers=num_layers_enc, num_heads=num_heads)
        self.cod_mapper = nn.Sequential(
            nn.Conv2d(c_hidden, c_codebook, kernel_size=1),
            nn.BatchNorm2d(c_codebook),
        )
        self.cod_unmapper = nn.Conv2d(c_codebook, c_hidden, kernel_size=1)
        self.decoder = Decoder(hidden_channels=c_hidden, size=latent_size, compressed_frames=compressed_frames, num_layers=num_layers_dec, num_heads=num_heads)

        self.codebook_size = codebook_size
        self.vqmodule = VQModule(
            c_codebook, k=codebook_size,
            q_init=0, q_refresh_step=15010, q_refresh_end=15010 * 130
            # q_init=15010 * 20, q_refresh_step=15010, q_refresh_end=15010 * 130
        )

    def encode(self, x):
        x = self.encoder(x)
        bs, tz, cz, hz, wz = x.shape
        x = x.view(bs*tz, cz, hz, wz)
        x = self.cod_mapper(x)
        qe, commit_loss, indices = self.vqmodule(x, dim=1)
        return (x, qe), commit_loss, indices

    def decode(self, x):
        x = self.cod_unmapper(x)
        x = x.view(self.batch_size, -1, self.c_hidden, self.latent_size, self.latent_size)
        x = self.decoder(x)
        return x

    def decode_indices(self, x):
        return self.decode(self.vqmodule.vquantizer.idx2vq(x, dim=1))

    def forward(self, x):
        print(x.shape)
        (_, qe), commit_loss, _ = self.encode(x)
        decoded = self.decode(qe)
        print(decoded.shape)
        return decoded, commit_loss


if __name__ == '__main__':
    e = Encoder(size=16)
    vq = VQModel(latent_size=16)
    print(sum([p.numel() for p in e.parameters()]))
    x = torch.randn(1, 100, 3, 128, 128)
    vq(x)