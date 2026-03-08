import torch
from einops import rearrange, repeat

# TODO: Try RoPE / AliBi
def sincos_1d(L, D, device, dtype):
    # 1d sinusoidal position encoding where element j of ith patch embedding is encoded as:
    # PE[i, 2j]   = sin(i / 10000^(2j/D))  # even indices
    # PE[i, 2j+1] = cos(i / 10000^(2j/D))  # odd indices

    assert D % 2 == 0, "Encoding dimension must be even"

    # position indices [L, 1] and dimension indices [1, D/2]
    pos = rearrange(torch.arange(L, device=device, dtype=dtype), 'l -> l 1')     # [L,1]
    i   = rearrange(torch.arange(D // 2, device=device, dtype=dtype), 'd -> 1 d')# [1,D/2]

    # angular frequencies: 1/10000^(2i/D) for each dimension
    div = torch.pow(torch.tensor(10000.0, device=device, dtype=dtype), (2*i)/D)

    # angles: pos * freq for each position-dimension pair
    angles = pos / div  # [L, D/2] (broadcasted together)
    pe = torch.zeros(L, D, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angles)  # even indices
    pe[:, 1::2] = torch.cos(angles)  # odd indices
    return pe # [L, D]

def sincos_time(T, D, device, dtype):
    # temporal PE (1d sinusoidal PE across time)
    return sincos_1d(T, D, device, dtype)  # reuse the same 1D builder


def build_spatial_only_pe(frame_size, patch_size, embed_dim, device='cpu', dtype=torch.float32):
    # spatial positional encodings for a grid of patches in first 2/3 of embed dim (evenly into x and y axes)
    # last 1/3 for temporal PE padded with 0s
    H, W = frame_size
    Hp, Wp = H // patch_size, W // patch_size

    # split dimensions (ensure temporal even)
    temporal_dim = (embed_dim // 3) & ~1
    spatial_dims = embed_dim - temporal_dim

    # split spatial dims between x and y (ensure both even)
    spatial_x_dim = (spatial_dims // 2) & ~1
    spatial_y_dim = spatial_dims - spatial_x_dim

    assert spatial_x_dim % 2 == 0 and spatial_y_dim % 2 == 0 and temporal_dim % 2 == 0

    # 2d PE for x and y axes
    pe_x = sincos_1d(Wp, spatial_x_dim, device, dtype)  # [Wp, Dx]
    pe_y = sincos_1d(Hp, spatial_y_dim, device, dtype)  # [Hp, Dy]
    pe_x = repeat(pe_x, 'wp dx -> hp wp dx', hp=Hp) # [Hp, Wp, Dx]
    pe_y = repeat(pe_y, 'hp dy -> hp wp dy', wp=Wp) # [Hp, Wp, Dy]

    pe_spatial = torch.cat([
        pe_x,
        pe_y,
        torch.zeros(Hp, Wp, temporal_dim, device=device, dtype=dtype)  # zero temporal tail
    ], dim=-1)  # [Hp, Wp, E]

    pe_spatial = rearrange(pe_spatial, 'hp wp e -> 1 (hp wp) e')  # [1, P, E]
    return pe_spatial  # [1, P, E]