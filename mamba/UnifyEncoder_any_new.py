import math
from functools import partial
from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

from utils.Decoder2 import TokenToImageDecoder

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

try:
    from mamba.csm_triton import cross_scan_fn, cross_merge_fn
except:
    from mamba.csm_triton import cross_scan_fn, cross_merge_fn

try:
    from mamba.csms6s import selective_scan_fn, selective_scan_flop_jit
except:
    from mamba.csms6s import selective_scan_fn, selective_scan_flop_jit

# FLOPs counter not prepared fro mamba2
try:
    from .mamba2.ssd_minimal import selective_scan_chunk_fn
except:
    from mamba2.ssd_minimal import selective_scan_chunk_fn


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))  # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))  # (K, inner)
        del dt_projs

        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class MambaBlock(nn.Module):
    def __init__(self, d_model=96, d_state=16, ssm_ratio=2.0, dt_rank="auto", dropout=0.0, scan_chan=False):
        super(MambaBlock, self).__init__()

        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_norm = nn.LayerNorm(d_model)
        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj, A, D ============================
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
        )

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.scan_chan = scan_chan

    def forward(self, x, seq=True, force_fp32=True):
        if self.scan_chan:
            x = x.permute(0, 2, 3, 1)
        x = x.permute(0, 2, 3, 1)
        x = self.in_proj(self.in_norm(x))  # [b, h, w, d_model] -> [b, h, w, d_inner * 2]
        x, z = x.chunk(2, dim=-1)  # x:[b, h, w, d_inner]   z:[b, h, w, d_inner]
        z = self.act(z)  # nn.SiLU
        x = x.permute(0, 3, 1, 2).contiguous()  # x: [b, d_inner, h, w]
        x = self.conv2d(x)  # x:[b, d_inner, h, w]
        x = self.act(x)  # nn.SiLU

        selective_scan = partial(selective_scan_fn, backend="mamba")

        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        x_hwwh = (torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1)
                  .view(B, 2, -1, L))
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        if hasattr(self, "x_proj_bias"):
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.contiguous()  # (b, k, d_state, l)
        Cs = Cs.contiguous()  # (b, k, d_state, l)

        As = -self.A_logs.float().exp()  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i],
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y

        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        y = y * z  # y: [b, h, w, d_model]
        out = self.dropout(self.out_proj(y)).permute(0, 3, 1, 2)  # out:[b, d_model, h, w]
        return out.permute(0, 3, 1, 2).contiguous() if self.scan_chan else out


class FeedForward(nn.Module):
    def __init__(self, inChannel, ffn_expansion_factor, firstCrossMambaInput, bias):
        super(FeedForward, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inChannel, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=firstCrossMambaInput, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(firstCrossMambaInput)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x




class DynamicFusionModule(nn.Module):
    def __init__(self, num_modals, input_channels, input_height, input_width, device):
        super(DynamicFusionModule, self).__init__()
        self.device = device
        self.num_modals = num_modals
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width

        self.modality_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
            ) for _ in range(num_modals)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(input_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, input_channels)
        )

    def forward(self, x):
        M = len(x)
        B, C, H, W = x[0].shape  

        modality_features = [x[i] for i in range(M)]
        fused_features = torch.zeros(B, C, H, W).to(self.device)
        for i in range(self.num_modals):
            processed = self.modality_processors[i](modality_features[i])
            fused_features = torch.add(fused_features, processed)
        fused_features = self.mlp(fused_features.permute(0, 2, 3, 1))

        projected_features = fused_features.permute(0, 3, 1,2)

        outputs = []
        for i in range(self.num_modals):
            original_feature = modality_features[i]
            added_back = original_feature + projected_features
            outputs.append(added_back)  

        output = torch.stack(outputs, dim=0)

        return output
class FusionMamba(nn.Module):
    def __init__(self, d_model=64, d_state=16, ssm_ratio=2.0, dt_rank="auto", dropout=0.0, modalities=2, patch_size=16):
        super(FusionMamba, self).__init__()
        self.patch_size = patch_size
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.modalities = modalities
        self.in_norm = nn.LayerNorm(d_model)

        # in proj ============================
        self.Proj = nn.ModuleList()
        for i in range(self.modalities):
            in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
            self.Proj.append(in_proj)

        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj, A, D ============================
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
        )

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, 768),
            nn.LayerNorm(768)
    )
    def selectiveScan(self, x, seq=True, force_fp32=True):
        selective_scan = partial(selective_scan_fn, backend="mamba")

        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        x_hwwh = (
            torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1)
            .view(B, 2, -1, L))
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        if hasattr(self, "x_proj_bias"):
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.contiguous()  # (b, k, d_state, l)
        Cs = Cs.contiguous()  # (b, k, d_state, l)

        As = -self.A_logs.float().exp()  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i],
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y

        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return y  # y:[b, h, w, d_inner]

    def forward(self, x):
        residualTemp = x
        residualTemp = torch.stack(residualTemp, dim=0)
        Proj = []
        Z = []
        for i, in_proj in enumerate(self.Proj):
            proj = x[i].permute(0, 2, 3, 1)
            proj = in_proj(self.in_norm(proj))
            proj, z = proj.chunk(2, dim=-1)
            z = self.act(z)
            proj = proj.permute(0, 3, 1, 2).contiguous()
            Proj.append(proj)
            Z.append(z)
        proj_cat = torch.stack(Proj, dim=0)
        z_cat = torch.stack(Z, dim=0)
        proj_cat = proj_cat.sum(dim=0, keepdim=True).squeeze(0)
        z_cat = z_cat.sum(dim=0, keepdim=True).squeeze(0)

        y = self.selectiveScan(proj_cat)

        y = y * z_cat
        x_cat = residualTemp.sum(dim=0, keepdim=True).squeeze(0)
        out = self.dropout(self.out_proj(y)).permute(0, 3, 1, 2)  # out:[b, d_model, h, w]
        fusion = x_cat + out

        fusion = fusion.permute(0, 2, 3, 1)
        fusion = fusion.reshape(len(fusion), -1, 64)

        # fusion = fusion.flatten(start_dim=1)
        x_transformed = self.fc(fusion)

        return x_transformed

class HierarchicalCompressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(121, 64, 3, stride=3),  # 121→40
            nn.ReLU()
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(64, 32, 5, stride=2),  # 40→18
            nn.ReLU()
        )
        self.stage3 = nn.Conv1d(32, 16, 3, stride=1)  # 18→16

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x.permute(0, 2, 1)

class QFormer(nn.Module):
    def __init__(self, dim=512, num_queries=16):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_queries, dim))  # [1, 16, 512]
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8)

    def forward(self, x):
        batch_size = x.shape[0]
        queries = self.query.repeat(batch_size, 1, 1)  # [batch_size, 16, 512]

        queries = queries.permute(1, 0, 2)  # [16, batch_size, 512]
        x_permuted = x.permute(1, 0, 2)  # [121, batch_size, 512]

        output, _ = self.attn(query=queries, key=x_permuted, value=x_permuted)

        output = output.permute(1, 0, 2)  # [batch_size, 16, 512]
        return output


class PoolCompressor(nn.Module):
    def __init__(self, outC):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(outC)  

    def forward(self, x):
        # x shape: [b, 121, 512]
        x = x.permute(0, 2, 1)  # [b, 512, 121]
        x = self.pool(x)        # [b, 512, 16]
        return x.permute(0, 2, 1)


class MultiModalModel(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels,
                            padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2)
        )
        return block


    def __init__(self, in_channels_list, pache_size, device):
        super(MultiModalModel, self).__init__()
        self.device = device
        self.pache_size = pache_size
        self.in_channels_list = in_channels_list
        self.Feedforward = nn.ModuleList()
        self.Branches_1 = nn.ModuleList()
        self.mlp_1 = DynamicFusionModule(len(in_channels_list), 96, self.pache_size, self.pache_size, self.device)
        self.share_mamba_1 = MambaBlock()
        self.Conv_1 = nn.ModuleList()
    
        self.Branches_2 = nn.ModuleList()
        self.mlp_2 = DynamicFusionModule(len(in_channels_list), 64, int(self.pache_size), int(self.pache_size),
                                         self.device)
        self.share_mamba_2 = MambaBlock(64)
        self.Conv_2 = nn.ModuleList()

        self.fusion = FusionMamba(64, modalities=len(in_channels_list), patch_size=self.pache_size)
        self.Specific = nn.ModuleList()
        self.Specific_prompt = nn.ModuleList()
        self.Decoders = nn.ModuleList()

        for in_channels in in_channels_list:
            IN_forward = FeedForward(in_channels, 2, 96, False)
            self.Feedforward.append(IN_forward)
            branch_1 = MambaBlock()
            self.Branches_1.append(branch_1)
            conv_1 = self.contracting_block(96, 64)
            self.Conv_1.append(conv_1)
            if in_channels == 19:
                branch_2 = MambaBlock(int(self.pache_size), scan_chan=True)
            else:
                branch_2 = MambaBlock(64)
            self.Branches_2.append(branch_2)
            #
            conv_2 = self.contracting_block(64, 64)
            self.Conv_2.append(conv_2)

            self.Specific.append(nn.Sequential(nn.Linear(768, 512)))

            self.Specific_prompt.append(PoolCompressor(16))
            # self.Specific_prompt.append(QFormer(dim=512, num_queries=16))

            self.Decoders.append(TokenToImageDecoder(512, [256, 128], in_channels))

    def forward(self, *inputs):

        if len(inputs) != len(self.Feedforward):
            raise ValueError("input error")

        forwards = []
        features_1 = []
        Features_1 = []

        features_2 = []
        Features_2 = []

        conv_1 = []
        conv_2 = []
        Spectial = []
        Spectial_prompt = []
        Spectial_align = []
        Decoders = []

        for i, in_forward in enumerate(self.Feedforward):
            x = in_forward(inputs[i])
            forwards.append(x)
        for i, branch in enumerate(self.Branches_1):
            x = branch(forwards[i])
            features_1.append(x)
        features_1 = self.mlp_1(features_1)
        for i in range(len(features_1)):
            Features_1.append(self.share_mamba_1(features_1[i]))
        for i, conv in enumerate(self.Conv_1):
            x = conv(Features_1[i])
            conv_1.append(x)
        for i, branch in enumerate(self.Branches_2):
            x = branch(conv_1[i])
            features_2.append(x)
        features_2 = self.mlp_2(features_2)
        for i in range(len(features_2)):
            Features_2.append(self.share_mamba_2(features_2[i]))
        for i, conv in enumerate(self.Conv_2):
            x = conv(Features_2[i])
            conv_2.append(x)

        fusion = self.fusion(conv_2)

        for i, proj_spe in enumerate(self.Specific):
            x = proj_spe(fusion)
            Spectial.append(x)
            Spectial_prompt.append(self.Specific_prompt[i](x))

        for i, decoder in enumerate(self.Decoders):
            x = decoder(Spectial[i],self.pache_size,self.pache_size)
            Decoders.append(x)

        return fusion, Spectial_prompt, Decoders, Spectial

def l1_loss(in_tensors, decoded_tensors):
    losses = []
    for input_tensor, decoded_tensor in zip(in_tensors, decoded_tensors):
        loss = F.l1_loss(decoded_tensor, input_tensor)
        losses.append(loss)
    return sum(losses) / len(losses)
