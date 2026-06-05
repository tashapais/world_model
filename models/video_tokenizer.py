from models.utils import ModelType
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from models.st_transformer import STTransformer
from models.fsq import FiniteScalarQuantizer
from models.patch_embed import PatchEmbedding
from models.positional_encoding import build_spatial_only_pe

class VideoTokenizerEncoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8,
                 hidden_dim=256, num_blocks=4, latent_dim=5, use_rope=False, use_adaln_zero=False):
        super().__init__()
        H, W = frame_size
        grid_size = (H // patch_size, W // patch_size)
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True,
                                         use_rope=use_rope, grid_size=grid_size,
                                         use_adaln_zero=use_adaln_zero)
        self.latent_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, latent_dim)
        )

    def forward(self, frames):
        # frames: [B, T, C, H, W]
        # frames to patch embeddings, pass through transformer, project to latent dim
        embeddings = self.patch_embed(frames)  # [B, T, P, E]
        transformed = self.transformer(embeddings) # [B, T, P, E]
        predicted_latents = self.latent_head(transformed) # [B, T, P, L]
        return predicted_latents


class PixelShuffleFrameHead(nn.Module):
    # conv2D embeddings to pixels head
    def __init__(self, embed_dim, patch_size=8, channels=3, H=128, W=128, use_conv_refiner=False):
        super().__init__()
        self.patch_size = patch_size
        self.Hp, self.Wp = H // patch_size, W // patch_size
        self.to_pixels = nn.Conv2d(embed_dim, channels * (patch_size ** 2), kernel_size=1)

        # Optional conv refiner: the 1x1 pixel-shuffle above decodes every patch
        # independently, which leaves visible patch-grid seams. A few full-res 3x3
        # convs mix across patch boundaries to remove that blockiness. Applied as a
        # residual with a zero-initialized last layer so it starts as identity.
        self.use_conv_refiner = use_conv_refiner
        if use_conv_refiner:
            self.refiner = nn.Sequential(
                nn.Conv2d(channels, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, channels, 3, padding=1),
            )
            nn.init.zeros_(self.refiner[-1].weight)
            nn.init.zeros_(self.refiner[-1].bias)

    def forward(self, tokens):  # [B, T, P, E]
        B, T, P, E = tokens.shape
        x = rearrange(tokens, 'b t (hp wp) e -> (b t) e hp wp', hp=self.Hp, wp=self.Wp) # [(B*T), E, Hp, Wp]
        x = self.to_pixels(x)                  # [(B*T), C*p^2, Hp, Wp]
        x = rearrange(x, '(b t) (c p1 p2) hp wp -> (b t) c (hp p1) (wp p2)', p1=self.patch_size, p2=self.patch_size, b=B, t=T) # [(B*T), C, H, W]
        if self.use_conv_refiner:
            x = x + self.refiner(x)            # residual seam-removal at full resolution
        x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T) # [B, T, C, H, W]
        return x


class VideoTokenizerDecoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8,
                 hidden_dim=256, num_blocks=4, latent_dim=5, use_rope=False, use_adaln_zero=False,
                 use_conv_refiner=False):
        super().__init__()
        H, W = frame_size
        self.patch_size = patch_size
        self.Hp, self.Wp = H // patch_size, W // patch_size
        self.num_patches = self.Hp * self.Wp
        self.use_rope = use_rope

        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True,
                                         use_rope=use_rope, grid_size=(self.Hp, self.Wp),
                                         use_adaln_zero=use_adaln_zero)
        self.frame_head = PixelShuffleFrameHead(embed_dim, patch_size=patch_size, channels=3, H=H, W=W,
                                                use_conv_refiner=use_conv_refiner)

        # spatial PE (only used when not using RoPE)
        pe_spatial_dec = build_spatial_only_pe((H, W), self.patch_size, embed_dim, device='cpu', dtype=torch.float32)  # [1,P,E]
        self.register_buffer("pos_spatial_dec", pe_spatial_dec, persistent=False)

    def forward(self, latents):
        # latents: [B, T, P, L]
        # embed latents and optionally add spatial PE
        embedding = self.latent_embed(latents)  # [B, T, P, E]
        if not self.use_rope:
            embedding = embedding + self.pos_spatial_dec.to(dtype=embedding.dtype, device=embedding.device)

        # apply transformer (temporal PE or RoPE applied inside)
        embedding = self.transformer(embedding)  # [B, T, P, E]

        # reconstruct frames using patch-wise head
        frames_out = self.frame_head(embedding)  # [B, T, C, H, W]

        return frames_out


class VideoTokenizer(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8,
                 hidden_dim=256, num_blocks=4, latent_dim=3, num_bins=4, use_rope=False, use_adaln_zero=False,
                 lpips_weight=0.0, lpips_net="vgg", use_conv_refiner=False):
        super().__init__()
        self.encoder = VideoTokenizerEncoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim, use_rope=use_rope, use_adaln_zero=use_adaln_zero)
        self.decoder = VideoTokenizerDecoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim, use_rope=use_rope, use_adaln_zero=use_adaln_zero, use_conv_refiner=use_conv_refiner)
        self.quantizer = FiniteScalarQuantizer(latent_dim, num_bins)
        self.codebook_size = num_bins**latent_dim

        # Perceptual (LPIPS) loss. smooth_l1 alone averages away high-frequency
        # detail -> blurry recons; LPIPS restores sharp edges/texture. The net is
        # held in a plain list (NOT a registered submodule) so its frozen VGG
        # weights stay out of state_dict/optimizer; call setup_lpips() before training.
        self.lpips_weight = float(lpips_weight)
        self._lpips_net_name = lpips_net
        self._lpips_holder = []

    def setup_lpips(self, device):
        if self.lpips_weight > 0 and not self._lpips_holder:
            import lpips as _lpips
            net = _lpips.LPIPS(net=self._lpips_net_name, verbose=False).to(device).eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self._lpips_holder.append(net)

    def forward(self, frames):
        # encode frames to latent representations, quantize, and decode back to frames
        embeddings = self.encoder(frames)  # [B, T, P, L]
        quantized_z = self.quantizer(embeddings)
        x_hat = self.decoder(quantized_z)  # [B, T, C, H, W]
        recon_loss = F.smooth_l1_loss(x_hat, frames)
        loss = recon_loss
        lpips_loss = torch.zeros((), device=frames.device, dtype=recon_loss.dtype)
        if self.lpips_weight > 0 and self._lpips_holder:
            B, T, C, H, W = frames.shape
            net = self._lpips_holder[0]
            # LPIPS expects [-1, 1]; the data pipeline (Normalize((0.5),(0.5)))
            # already delivers frames in [-1, 1], so pass through directly.
            xr = x_hat.reshape(B * T, C, H, W)
            xt = frames.reshape(B * T, C, H, W)
            lpips_loss = net(xr, xt).mean()
            loss = recon_loss + self.lpips_weight * lpips_loss
        logs = {'recon': recon_loss.detach(), 'lpips': lpips_loss.detach()}
        return loss, x_hat, logs

    def tokenize(self, frames):
        # encode frames to latent representations, quantize, and return indices
        embeddings = self.encoder(frames)  # [B, T, P, L]
        quantized_z = self.quantizer(embeddings)
        indices = self.quantizer.get_indices_from_latents(quantized_z, dim=-1)
        return indices

    def detokenize(self, quantized_z):
        # decode quantized latents back to frames
        x_hat = self.decoder(quantized_z)  # [B, T, C, H, W]
        return x_hat

    @property
    def model_type(self) -> str:
        return ModelType.VideoTokenizer