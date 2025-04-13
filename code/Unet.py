import torch
import torch.nn as nn
from functools import partial
import math
from einops import rearrange
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))


def DownSample(dim):
    return nn.Conv2d(dim, dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=(1, 1), bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=(1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.scale = dim_head ** -0.5
        self.head = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=(1, 1), bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=(1, 1)),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(chunk=3, dim=1)  # 在第一维度，将tensor一分为三
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)  # 进行广义张量乘积

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Residual(nn.Module):
    def __init__(self, dim, fn):
        super(Residual, self).__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x_ = self.norm(x)

        return self.fn(x_) + x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=4):
        super(ResnetBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None
        self.block1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )  # Identity 输入=输出
        self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=(1, 1)) if dim != dim_out else nn.Identity()

    # 先卷一次，再+time_emb,再卷一次，再+1x1，然后输出
    def forward(self, inputs, time_emb=None):
        h = self.block1(inputs)
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)  # [128, 112] -> [128, dim_out]
            h = rearrange(time_emb, 'b c -> b c 1 1') + h

        h = self.block2(h)
        return h + self.res_conv(inputs)


#  对t进行位置编码
class SinusoidalPosEmb(nn.Module):
    def __init__(self, in_dim=28):
        super(SinusoidalPosEmb, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):  # [128]
        device = x.device
        half_dim = self.in_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb  # [128, 28]


class UNet(nn.Module):
    def __init__(self,
                 img_size,

                 channels,
                 out_dim=None,
                 dim_mults=(1, 2, 4, 8),
                 resnet_block_groups=4,
                 with_time_emb=True,
                 learned_variance=False
                 ):
        super(UNet, self).__init__()
        self.channels = channels  # 1
        init_dim = 18
        self.init_conv = nn.Conv2d(channels, init_dim, kernel_size=(7, 7), padding=(3, 3))
        dims = [init_dim, *map(lambda m: img_size * m, dim_mults)]  # [18, 28, 56, 112]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)  # 固定参数groups=resnet_block_groups，并返回新函数

        # 对t_list进行位置编码
        if with_time_emb:
            time_dim = img_size * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(in_dim=img_size),
                nn.Linear(img_size, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # conv layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolution = len(in_out)  # 大模块数量
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= num_resolution - 1  # 判断是否是最后一组数据

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(dim=dim_out, fn=LinearAttention(dim_out)),
                DownSample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(mid_dim, Attention(mid_dim))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= num_resolution - 1

            self.ups.append(nn.ModuleList([
                block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(dim_in, LinearAttention(dim_in)),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_conv = nn.Sequential(
            block_klass(img_size, img_size),
            nn.Conv2d(img_size, self.out_dim, kernel_size=(1, 1))
        )

    def forward(self, sample_list, time_list):
        x = self.init_conv(sample_list)
        t = self.time_mlp(time_list) if exists(self.time_mlp) else None

        h = []

        # DownSample
        for block1, block2, attn, downSample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downSample(x)

        # bottleNeck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upSample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


if __name__ == '__main__':
    pass
