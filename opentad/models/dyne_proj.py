import torch
import torch.nn as nn
import torch.nn.functional as F

from .actionformer_proj import get_sinusoid_encoding
from .bricks import ConvModule, AffineDropPath
from .builder import PROJECTIONS


@PROJECTIONS.register_module()
class DynEProj(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
        use_abs_pe=False,  # use absolute position embedding
        max_seq_len=2304,
        mlp_dim=512,  # the number of dim of mlp
        encoder_win_size=1,  # size of local window for mha
        k=1.5,  # the expanded kernel weight
        init_conv_vars=1.0,  # initialization of gaussian variance for the weight
        path_pdrop=0.0,  # dropout rate for drop path
        input_noise=0.0,
    ):
        super().__init__()
        assert len(arch) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = arch
        self.use_abs_pe = use_abs_pe
        self.max_seq_len = max_seq_len
        self.input_noise = input_noise

        if isinstance(self.in_channels, (list, tuple)):
            assert isinstance(self.out_channels, (list, tuple)) and len(self.in_channels) == len(self.out_channels)
            self.proj_layers = nn.ModuleList([])
            for n_in, n_out in zip(self.in_channels, self.out_channels):
                self.proj_layers.append(
                    ConvModule(
                        n_in,
                        n_out,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            in_channels = out_channels = sum(self.out_channels)
        else:
            self.proj_layers = None

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embed)
        if self.use_abs_pe:
            pos_embed = get_sinusoid_encoding(self.max_seq_len, out_channels) / (out_channels**0.5)
            self.register_buffer("pos_embed", pos_embed, persistent=False)

        # embedding network using convs
        self.embed_blocks = nn.ModuleList()
        for i in range(arch[0]):
            self.embed_blocks.append(
                ConvModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )

        # stem network using (vanilla) transformer
        self.stem_blocks = nn.ModuleList()
        for _ in range(arch[1]):
            self.stem_blocks.append(
                GDKLayer(out_channels, 1, downsample_stride=1, mlp_hidden_dim=mlp_dim, k=k, init_conv_vars=init_conv_vars)
            )

        # main branch using transformer with pooling
        self.branch_blocks = nn.ModuleList()
        for _ in range(arch[2]):
            self.branch_blocks.append(
                GDKLayer(
                    out_channels,
                    encoder_win_size,
                    downsample_stride=2,
                    path_pdrop=path_pdrop,
                    mlp_hidden_dim=mlp_dim,
                    k=k,
                    init_conv_vars=init_conv_vars,
                )
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, sequence length (bool)

        # trick, adding noise may slightly increases the variability between input features.
        if self.training and self.input_noise > 0:
            noise = torch.randn_like(x) * self.input_noise
            x += noise

        # feature projection
        if self.proj_layers is not None:
            x = torch.cat([proj(s, mask)[0] for proj, s in zip(self.proj_layers, x.split(self.in_channels, dim=1))], dim=1)

        # embedding network
        for block in self.embed_blocks:
            x, mask = block(x, mask)

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert x.shape[-1] <= self.max_seq_len, "Reached max length."
            pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if x.shape[-1] >= self.max_seq_len:
                pe = F.interpolate(self.pos_embed, x.shape[-1], mode="linear", align_corners=False)
            else:
                pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # stem transformer
        for block in self.stem_blocks:
            x, mask = block(x, mask)

        # prep for outputs
        out_feats = (x,)
        out_masks = (mask,)

        # main branch with downsampling
        for block in self.branch_blocks:
            x, mask = block(x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks


class GDKLayer(nn.Module):
    def __init__(
        self,
        embed_dim,  # dimension of the input features
        kernel_size=3,  # conv kernel size
        downsample_stride=1,  # downsampling stride for the current layer
        k=1.5,  # expansion factor for adaptive kernel
        group=1,  # group for cnn
        output_dim=None,  # output dimension, if None, set to input dim
        mlp_hidden_dim=None,  # hidden dim for mlp
        path_pdrop=0.0,  # drop path rate
        act_layer=nn.GELU,  # nonlinear activation used in mlp,
        init_conv_vars=0.1,  # init gaussian variance for the weight
        variable_kernel_sizes=None,  # candidate kernel sizes for adaptive convkw
        gate_hidden_ratio=0.25,
        gate_temperature=1.0,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.downsample_stride = downsample_stride
        self.gate_temperature = gate_temperature
        if output_dim is None:
            output_dim = embed_dim

        # 新增：主分支和MLP分支前后都加一组LayerNorm和GroupNorm
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        # self.gn1 = nn.GroupNorm(16, embed_dim, eps=1e-6)
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
        # self.gn2 = nn.GroupNorm(16, embed_dim, eps=1e-6)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        upsampled_kernel_size = round((kernel_size + 1) * k)
        upsampled_kernel_size = upsampled_kernel_size + 1 if upsampled_kernel_size % 2 == 0 else upsampled_kernel_size

        self.psi_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size, stride=1, padding=kernel_size // 2, groups=embed_dim)
        self.weight_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size, stride=1, padding=kernel_size // 2, groups=embed_dim)
        
        if variable_kernel_sizes is None:
            variable_kernel_sizes = (kernel_size, upsampled_kernel_size)
        adaptive_kernel_list = []
        for ksz in variable_kernel_sizes:
            ksz = int(round(ksz))
            if ksz % 2 == 0:
                ksz += 1
            adaptive_kernel_list.append(max(3, ksz))
        adaptive_kernel_list = sorted(set(adaptive_kernel_list))
        self.adaptive_kernel_sizes = tuple(adaptive_kernel_list)
        self.adaptive_conv_branches = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, embed_dim, ksz, stride=1, padding=ksz // 2, groups=embed_dim)
                for ksz in self.adaptive_kernel_sizes
            ]
        )
        self.num_kernels = len(self.adaptive_kernel_sizes)

        gate_hidden_dim = max(int(embed_dim * gate_hidden_ratio), 16)
        self.kernel_gate_net = nn.Sequential(
            nn.Conv1d(embed_dim, gate_hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(gate_hidden_dim, self.num_kernels, 1),
        )
        if downsample_stride > 1:
            kernel_size_pool, stride_pool, padding_pool = downsample_stride + 1, downsample_stride, (downsample_stride + 1) // 2
            self.downsample = nn.MaxPool1d(kernel_size_pool, stride=stride_pool, padding=padding_pool)
            self.stride = stride_pool
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        # two layer mlp
        if mlp_hidden_dim is None:
            mlp_hidden_dim = 4 * embed_dim  # default
        if output_dim is None:
            output_dim = embed_dim

        self.mlp_block = nn.Sequential(
            nn.Conv1d(embed_dim, mlp_hidden_dim, 1, groups=group),
            act_layer(),
            nn.Conv1d(mlp_hidden_dim, output_dim, 1, groups=group),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_main = AffineDropPath(embed_dim, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(output_dim, drop_prob=path_pdrop)
        else:
            self.drop_path_main = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.activation = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars, embed_dim=embed_dim)

    def reset_params(self, init_conv_vars=0, embed_dim=0):
        torch.nn.init.normal_(self.psi_conv.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.weight_conv.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi_conv.bias, 0)
        torch.nn.init.constant_(self.weight_conv.bias, 0)
        for conv_layer in self.adaptive_conv_branches:
            torch.nn.init.normal_(conv_layer.weight, 0, init_conv_vars)
            torch.nn.init.constant_(conv_layer.bias, 0)
        for module in self.kernel_gate_net:
            if isinstance(module, nn.Conv1d) and module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x, mask):
        # X shape: B, C, T
        B, C, T = x.shape
        x = self.downsample(x)
        out_mask = self.downsample(mask.unsqueeze(1).to(x.dtype)).detach()

        # 主分支输入归一化：先LayerNorm再GroupNorm
        normalized_x = self.ln1(x.permute(0, 2, 1)).permute(0, 2, 1)
        # normalized_x = self.gn1(normalized_x)

        psi_feat = self.psi_conv(normalized_x)
        weight_feat = self.weight_conv(normalized_x)

        adaptive_outputs = []
        for conv in self.adaptive_conv_branches:
            adaptive_outputs.append(conv(normalized_x))
        adaptive_stack = torch.stack(adaptive_outputs, dim=1)  # B, num_kernels, C, T

        mask_float = out_mask.to(normalized_x.dtype)
        if mask_float.dim() == 2:
            mask_float = mask_float.unsqueeze(1)
        valid_tokens = mask_float.sum(dim=-1, keepdim=True).clamp(min=1.0)
        global_context = (normalized_x * mask_float).sum(dim=-1, keepdim=True) / valid_tokens
        gate_logits = self.kernel_gate_net(global_context).squeeze(-1)
        gate_weights = torch.softmax(gate_logits / self.gate_temperature, dim=1)
        adaptive_fused = (gate_weights.view(gate_weights.shape[0], self.num_kernels, 1, 1) * adaptive_stack).sum(dim=1)

        main_out = torch.relu(weight_feat + adaptive_fused) * psi_feat
        main_out = x * out_mask + self.drop_path_main(main_out)

        # MLP分支输入归一化：先LayerNorm再GroupNorm
        mlp_input = self.ln2(main_out.permute(0, 2, 1)).permute(0, 2, 1)
        # mlp_input = self.gn2(mlp_input)
        final_out = main_out + self.drop_path_mlp(self.mlp_block(mlp_input))
        return final_out, out_mask.squeeze(1).bool()