# the code from https://github.com/lucidrains/video-diffusion-pytorch
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from einops import rearrange
from einops_exts import rearrange_many

from rotary_embedding_torch import RotaryEmbedding

from model.DM.text import BERT_MODEL_DIM


# helpers functions
def exists(x):
    return x is not None


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


# relative positional bias
class RelativePositionBias(nn.Module):
    def __init__(
            self,
            heads=8,
            num_buckets=32,
            max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim, use_deconv=True, padding_mode="reflect"):
    if use_deconv:
        return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))
    else:
        return nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'),
            nn.Conv3d(dim, dim, (1, 3, 3), (1, 1, 1), (0, 1, 1), padding_mode=padding_mode)
        )


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

# building block modules
class Block_cond(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.tem_proj = nn.Conv3d(dim, dim, (3, 1, 1), padding=(1, 0, 0))
        self.spa_proj = nn.Conv3d(dim, dim, (1, 3, 3), padding=(0, 1, 1))
        self.norm1 = nn.GroupNorm(groups, dim)

        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        # self.norm2 = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 5)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1, 1)
        return feat_mean, feat_std


    def adaptive_instance_normalization(self, content_feat, style_feat):
        content_feat = content_feat.permute(0,2,1,3,4).contiguous()
        style_feat = style_feat.permute(0,2,1,3,4).contiguous()
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return (normalized_feat * style_std.expand(size) + style_mean.expand(size)).permute(0,2,1,3,4).contiguous()

    def forward(self, x, guidance=None):
        # res = x
        if x.shape[-2:]!=guidance.shape[-2:]:
            guidance = F.interpolate(guidance, size=x.shape[-3:], mode='trilinear')
        x = torch.cat([x, guidance], dim=1)
        x = self.spa_proj(x)
        x = self.tem_proj(x)
        x = self.norm1(x)

        x = self.proj(x)
        x = self.adaptive_instance_normalization(x, guidance)

        return self.act(x)

# building block modules
class Block_tem(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (3, 1, 1), padding=(1, 0, 0))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, guidance=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class ResnetBlock_w_Motion(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, motion_dim=0, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block_cond(dim_out + motion_dim, dim_out, groups=groups)
        self.block3 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, x_cond=None, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h, x_cond)
        h = self.block3(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)


# attention along space and time

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y

class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):        
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y

class MotionAdaptor(nn.Module):
    def __init__(self, channel_in, channel_hid, channel_out, N_T=4, incep_ker = [3,5,7,11], groups=8):
        super(MotionAdaptor, self).__init__()

        self.N_T = N_T
        
        enc_layers = []
        
        # channel in->channel hid
        enc_layers.append(Inception(channel_in, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        
        # channel hid->channel hid
        for _ in range(1, N_T):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))

        dec_layers = []
        
        # channel hid->channel hid
        dec_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        
        # channel hid + skip -> channel hid
        for _ in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
            
        # channel hid + skip -> channel output
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_out, incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> b (t c) h w')

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = rearrange(z, 'b (t c) h w -> b c t h w', c=C)
        return y

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
            self,
            x,
            pos_bias=None,
            focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


# model
class Unet3D(nn.Module):
    def __init__(
            self,
            dim,
            cond_dim=None,
            out_grid_dim=2,
            out_conf_dim=1,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            cond_channels=3,
            attn_heads=8,
            attn_dim_head=32,
            use_bert_text_cond=False,
            init_dim=None,
            init_kernel_size=7,
            use_sparse_linear_attn=True,
            resnet_groups=8,
            use_final_activation=False,
            learn_null_cond=False,
            use_deconv=True,
            padding_mode="zeros",
            cond_num=0,
            pred_num=0,
    ):
        self.tc = cond_num
        self.tp = pred_num
        
        super().__init__()
        self.null_cond_mask = None
        self.channels = channels

        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        
        # TODO: 
        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c',
                                                    Attention(dim, heads=attn_heads, dim_head=attn_dim_head,
                                                              rotary_emb=rotary_emb))

        self.time_rel_pos_bias = RelativePositionBias(heads=attn_heads,
                                                      max_distance=32)  # realistically will not be able to generate that many frames of video... yet

        # initial conv
        init_dim = default(init_dim, dim)
        motion_dim = 16
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        
        self.init_conv = nn.Conv3d(channels, init_dim, kernel_size=(1, init_kernel_size, init_kernel_size), padding=(0, init_padding, init_padding))
        
        self.init_cond_conv = nn.Conv3d(cond_channels, motion_dim, kernel_size=(1, init_kernel_size, init_kernel_size), padding=(0, init_padding, init_padding))
        
        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))
        self.init_cond_temporal_attn = Residual(PreNorm(motion_dim, temporal_attn(motion_dim)))

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # time conditioning
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning
        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim

        # modified by nhm
        self.learn_null_cond = learn_null_cond
        if self.learn_null_cond:
            self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim)) if self.has_cond else None
        else:
            self.null_cond_emb = torch.zeros(1, cond_dim).cuda() if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.motion_enc = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type
        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_motion_cond = partial(ResnetBlock_w_Motion, groups=resnet_groups, time_emb_dim=cond_dim)
        
        self.motion_enc.append(nn.ModuleList([
            block_klass(motion_dim, motion_dim),
            block_klass(motion_dim, motion_dim),
            Residual(PreNorm(motion_dim, SpatialLinearAttention(motion_dim, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
            Residual(PreNorm(motion_dim, temporal_attn(motion_dim))),
            MotionAdaptor(motion_dim*self.tc, 256, motion_dim*self.tp)
        ]))
        
        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            
            self.downs.append(nn.ModuleList([
                block_klass_motion_cond(dim_in, dim_out, motion_dim=motion_dim),
                block_klass_motion_cond(dim_out, dim_out, motion_dim=motion_dim),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_motion_cond(mid_dim, mid_dim, motion_dim=motion_dim)

        # TODO:
        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn =  Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_motion_cond(mid_dim, mid_dim, motion_dim=motion_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_motion_cond(dim_out * 2, dim_in, motion_dim=motion_dim),
                block_klass_motion_cond(dim_in, dim_in, motion_dim=motion_dim),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in, use_deconv, padding_mode) if not is_last else nn.Identity()
            ]))
            
        
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_grid_dim, 1)
        )

        # added by nhm
        self.use_final_activation = use_final_activation
        
        if self.use_final_activation:
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Identity()

        # added by nhm for predicting occlusion mask
        self.occlusion_map = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_conf_dim, 1)
        )

    def forward_with_cond_scale(self, *args, cond_scale=2.,  **kwargs ):
        if cond_scale == 0:
            null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
            return null_logits

        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(self, x, time, cond_frames, cond_fea=None, cond=None, null_cond_prob=0., none_cond_mask=None, focus_present_mask=None, prob_focus_present=0.):
        # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device
        tc, tp = cond_frames.shape[2], x.shape[2]
        assert tc == self.tc
        assert tp == self.tp
        
        # b c t h w
        # x = torch.cat([x, cond_frames], dim=2)
        if not cond_fea is None:
            x = torch.cat([x, cond_fea.unsqueeze(2).repeat(1,1,x.shape[2],1,1)], dim=1)
        
        focus_present_mask  = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device=device))

        time_rel_pos_bias   = self.time_rel_pos_bias(tp, device=x.device)
        time_cond_rel_pos_bias = self.time_rel_pos_bias(tc, device=x.device)

        x = self.init_conv(x)        
        r = x.clone()
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        
        # Motion Encoding
        cond_frames = self.init_cond_conv(cond_frames)
        cond_frames = self.init_cond_temporal_attn(cond_frames, pos_bias=time_cond_rel_pos_bias)

        for block1, block2, spatial_attn, temporal_attn, adaptor in self.motion_enc:
            cond_frames = block1(cond_frames)
            cond_frames = block2(cond_frames)
            cond_frames = spatial_attn(cond_frames)
            cond_frames = temporal_attn(cond_frames, pos_bias=time_cond_rel_pos_bias, focus_present_mask=focus_present_mask)
            cond_frames = adaptor(cond_frames)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance

        if self.has_cond:
            batch, device = x.shape[0], x.device
            self.null_cond_mask = prob_mask_like((batch,), null_cond_prob, device=device)
            if none_cond_mask is not None:
                self.null_cond_mask = torch.logical_or(self.null_cond_mask, torch.tensor(none_cond_mask).cuda())
            cond = torch.where(rearrange(self.null_cond_mask, 'b -> b 1'), self.null_cond_emb, cond)
            t = torch.cat((t, cond), dim=-1)

        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, cond_frames, t)
            x = block2(x, cond_frames, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, cond_frames, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, cond_frames, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, cond_frames, t)
            x = block2(x, cond_frames, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        _, x_fin = self.final_conv(x)
        _, x_occ = self.occlusion_map(x)
        return torch.cat((x_fin, x_occ), dim=1)