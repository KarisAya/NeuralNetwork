import torch


class FeedForward(torch.nn.Module):

    def __init__(self, emb_size: int):
        super().__init__()
        self.lm1 = torch.nn.Linear(emb_size, 4 * emb_size)
        self.lm2 = torch.nn.Linear(4 * emb_size, emb_size)
        self.dp = torch.nn.Dropout(0.4)

    def forward(self, x: torch.Tensor):
        out = torch.nn.functional.gelu(self.lm1(x))
        out = self.dp(self.lm2(out))
        return out


class SelfMultiHeadAttention(torch.nn.Module):
    def __init__(self, emb_size: int, head_size: int):
        assert emb_size % head_size == 0
        super().__init__()
        self.qkv_linear = torch.nn.Linear(emb_size, emb_size * 3, bias=False)
        self.proj = torch.nn.Linear(emb_size, emb_size)
        self.dp = torch.nn.Dropout(0.4)
        # 其他参数
        self.n_head = emb_size // head_size
        self.head_size = head_size
        self.scale = self.head_size**-0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        q, k, v = self.qkv_linear.forward(x).chunk(3, dim=-1)
        # 把 q, v 的形状从 (B, T, emb_size) 变为 (B, n_head, T, head_size)
        # 把 k 的形状从 (B, T, emb_size) 变为 (B, n_head, head_size, T)
        q = q.unflatten(dim=-1, sizes=(self.n_head, self.head_size)).transpose(-3, -2)
        k = k.transpose(-2, -1).unflatten(dim=-2, sizes=(self.n_head, self.head_size))
        v = v.unflatten(dim=-1, sizes=(self.n_head, self.head_size)).transpose(-3, -2)
        scores = (q @ k) * self.scale
        scores = scores.masked_fill(mask, float("-inf"))
        w_att = torch.softmax(scores, dim=-1)
        # 此时 attn_output 的形状为 (B, n_head, T, head_size)
        attn_output = w_att @ v
        # 恢复 attn_output 的形状为 (B, T, emb_size)
        attn_output = attn_output.transpose(-3, -2).flatten(start_dim=-2)
        mha = self.dp(self.proj(attn_output))
        return mha


class CrossMultiHeadAttention(torch.nn.Module):
    def __init__(self, emb_size: int, head_size: int):
        assert emb_size % head_size == 0
        super().__init__()
        self.kv_linear = torch.nn.Linear(emb_size, emb_size * 2, bias=False)
        self.q_linear = torch.nn.Linear(emb_size, emb_size, bias=False)
        self.proj = torch.nn.Linear(emb_size, emb_size)
        self.dp = torch.nn.Dropout(0.4)
        # 其他参数
        self.n_head = emb_size // head_size
        self.head_size = head_size
        self.scale = self.head_size**-0.5

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        k, v = self.kv_linear.forward(y).chunk(2, dim=-1)
        q = self.q_linear.forward(x)
        q = q.unflatten(dim=-1, sizes=(self.n_head, self.head_size)).transpose(-3, -2)
        k = k.transpose(-2, -1).unflatten(dim=-2, sizes=(self.n_head, self.head_size))
        v = v.unflatten(dim=-1, sizes=(self.n_head, self.head_size)).transpose(-3, -2)
        scores = (q @ k) * self.scale
        scores = scores.masked_fill(mask, float("-inf"))
        w_att = torch.softmax(scores, dim=-1)
        attn_output = w_att @ v
        attn_output = attn_output.transpose(-3, -2).flatten(start_dim=-2)
        mha = self.dp(self.proj(attn_output))
        return mha


class EncoderBlock(torch.nn.Module):
    def __init__(self, emb_size: int, head_size: int):
        super().__init__()
        self.mha_ln = torch.nn.LayerNorm(emb_size)
        self.mha = SelfMultiHeadAttention(emb_size, head_size)
        self.ff_ln = torch.nn.LayerNorm(emb_size)
        self.ff = FeedForward(emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = x + self.mha(self.mha_ln(x), mask)
        x = x + self.ff(self.ff_ln(x))
        return x


class DecoderBlock(torch.nn.Module):
    def __init__(self, emb_size: int, head_size: int):
        super().__init__()
        self.mha_ln = torch.nn.LayerNorm(emb_size)
        self.mha = SelfMultiHeadAttention(emb_size, head_size)
        self.cross_mha_ln = torch.nn.LayerNorm(emb_size)
        self.cross_mha = CrossMultiHeadAttention(emb_size, head_size)
        self.ff_ln = torch.nn.LayerNorm(emb_size)
        self.ff = FeedForward(emb_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor, self_mask: torch.Tensor, cross_mask: torch.Tensor):
        x = x + self.mha(self.mha_ln(x), self_mask)
        x = x + self.cross_mha(self.cross_mha_ln(x), y, cross_mask)
        x = x + self.ff(self.ff_ln(x))
        return x


class TransformerEDM(torch.nn.Module):
    def __init__(
        self,
        enc_vs: int,
        dec_vs: int,
        emb_size: int,
        head_size: int,
        n_block: int,
        sequence_length: int,
        padding_idx: int = 0,
    ):
        super().__init__()
        # 编码器
        self.enc_blocks = torch.nn.ModuleList()
        # 解码器
        self.dec_blocks = torch.nn.ModuleList()
        self.blocks = [(EncoderBlock(emb_size, head_size), DecoderBlock(emb_size, head_size)) for _ in range(n_block)]
        for enc_block, dec_block in self.blocks:
            self.enc_blocks.append(enc_block)
            self.dec_blocks.append(dec_block)
        self.ln = torch.nn.LayerNorm(emb_size)
        # 编码器词嵌入
        self.enc_token_emb = torch.nn.Embedding(enc_vs, emb_size)
        self.enc_position_emb = torch.nn.Embedding(sequence_length, emb_size)
        # 解码器词嵌入
        self.dec_token_emb = torch.nn.Embedding(dec_vs, emb_size)
        self.dec_position_emb = torch.nn.Embedding(sequence_length, emb_size)
        # 解码器输出层
        self.dec_lm = torch.nn.Linear(emb_size, dec_vs)

        # 下面是其他属性
        self.mask: torch.Tensor
        self.register_buffer("mask", ~torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool)))
        self.position: torch.Tensor
        self.register_buffer("position", torch.arange(0, sequence_length, dtype=torch.long))
        self.padding_idx = padding_idx

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        Tx = x.size(-1)
        Ty = y.size(-1)
        enc_inputs: torch.Tensor = self.enc_token_emb(x) + self.enc_position_emb(self.position[:Tx])
        dec_inputs: torch.Tensor = self.dec_token_emb(y) + self.dec_position_emb(self.position[:Ty])
        enc_padding_mask = (x == self.padding_idx).unsqueeze(-2)
        dec_padding_mask = (y == self.padding_idx).unsqueeze(-1)
        dec_cross_mask = dec_padding_mask | enc_padding_mask
        for enc_block, dec_block in self.blocks:
            enc_inputs = enc_block(enc_inputs, mask=enc_padding_mask.unsqueeze(-1))
            dec_inputs = dec_block(dec_inputs, enc_inputs, dec_padding_mask | self.mask[:Ty, :Ty], dec_cross_mask.unsqueeze(-3))
        out = self.dec_lm(self.ln(dec_inputs))
        return out
