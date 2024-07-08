import torch
import torch.nn as nn
import math
import einops
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from utils import get_main_parser
import numpy as np

# Get parser and parse arguments
parser = get_main_parser()
args = parser.parse_args()
args_dict = vars(args)

#Seed
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class Attention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, G = x.shape 
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, G // self.num_heads)
        qkv = einops.rearrange(qkv, 'b n k h d -> k b h n d')  
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) (h c fph)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, G)  # Cambiado de .transpose(0, 1) a .transpose(1, 2)
        #x = einops.rearrange(x, 'c h fph -> c (h fph)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
def modulate(x, shift, scale):
    res = x * (1 + scale) + shift
    return res

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    time emb to frequency_embedding_size dim, then to hidden_size
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
   
class DiTblock(nn.Module):
    # adaBN -> attn -> mlp
    def __init__(self,
                 feature_dim=2000,
                 mlp_ratio=4.0,
                 num_heads=10,
                 **kwargs) -> None:
        super().__init__()
        
        self.norm1 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention2(feature_dim, num_heads=num_heads, qkv_bias=True, **kwargs) 
        
        self.norm2 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU()
        
        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = Mlp(in_features=feature_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feature_dim, 6 * feature_dim, bias=True)
        )
    
    def forward(self,x,c):
        # Project condition to 6 * hiddensize and then slice it into 6 parts along the column.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1) #dim cambiado de 1 a 2
        # attention blk
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # mlp
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT. adaLN -> linear
    """
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Projected into output shape
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        # shift scale
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # shift scale
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1) #dim cambiado de 1 a 2
        x = modulate(self.norm_final(x), shift, scale)
        # projection
        x = self.linear(x)
        return x      

BaseBlock = {'dit':DiTblock}

class DiT_stDiff(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 depth,
                 dit_type,
                 num_heads,
                 classes,
                 mlp_ratio=4.0,
                 patch_size = 2, 
                 in_channels = 3,
                 **kwargs) -> None:
        """ denoising model

        Args:
            input_size (_type_): input dim
            hidden_size (_type_): scale input to hidden dim
            depth (_type_): dit block num
            dit_type (_type_): which type block to use
            num_heads (_type_): transformer head num
            classes (_type_): class num
            mlp_ratio (float, optional): _description_. Defaults to 4.0.
        """        
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.classes = classes
        self.mlp_ratio = mlp_ratio
        self.dit_type = dit_type
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
                
        #MODIFIED FROM nn.Linear to PatchEmbed
        self.in_layer = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        num_patches = self.in_layer.num_patches
        
        #MODIFIED FROM nn.Linear to PatchEmbed
        self.cond_layer = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)

        # celltype emb
        self.condi_emb = nn.Embedding(classes, hidden_size)

        # time emb
        self.time_emb = TimestepEmbedder(hidden_size=self.hidden_size)

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        # DiT block
        self.blks = nn.ModuleList([
            BaseBlock[dit_type](self.hidden_size, mlp_ratio=self.mlp_ratio, num_heads=self.num_heads) for _ in
            range(self.depth)
        ])

        # out layer
        self.out_layer = FinalLayer(self.hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                # xavier 
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # bias  0
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.in_layer.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.in_layer.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.in_layer.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.condi_emb.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_emb.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.dit_type == 'dit':
            for block in self.blks:
                # adaLN_modulation , adaLN  0 initial
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.out_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.linear.weight, 0)
        nn.init.constant_(self.out_layer.linear.bias, 0)
        
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.in_layer.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y,**kwargs): 
        x = x.float()
        x = self.in_layer(x) + self.pos_embed
        t = self.time_emb(t)
        y = self.cond_layer(y)
        c = t + y

        for blk in self.blks:
            x = blk(x, c)
        
        x = self.out_layer(x, c)
        x = self.unpatchify(x)
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
        