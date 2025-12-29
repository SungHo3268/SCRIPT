# srcs/generic_script_injection.py
import math
from typing import Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenericCrossAttention(nn.Module):
    """
    Architecture-agnostic cross attention:
      - Query: original hidden states (B, Tq, D)
      - Key/Value: script hidden states   (B, Tk, D)
    Notes
      * No positional embedding, no norm inside (요구사항)
      * Optional gated residual to keep stability at init
      * LoRA 타깃팅을 위해 q_proj/k_proj/v_proj/o_proj 이름 사용
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        use_bias: bool = True,
        gated_residual: bool = True,
        is_causal: bool = True,         # 필요 시 causal 마스크 사용
        layer_scale_init: float = 0.0,  # >0이면 layer-scale 적용 (안정화 옵션)
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        # self.head_dim = hidden_size // num_heads
        self.head_dim = intermediate_size // num_heads
        self.is_causal = is_causal

        # LoRA-friendly naming
        self.q_proj = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.k_proj = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.v_proj = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.o_proj = nn.Linear(intermediate_size, hidden_size, bias=use_bias)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)

        self.gated_residual = gated_residual
        if gated_residual:
            # 초기 0 → 거의 identity. 학습되며 script 기여도 점진적 증가
            self.gate = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("gate", None)

        # if layer_scale_init > 0.0:
        #     self.gamma = nn.Parameter(torch.full((hidden_size,), layer_scale_init))
        # else:
        #     self.gamma = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, H, T, Dh)
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, T, Dh) -> (B, T, D)
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def _causal_mask(self, q_len: int, k_len: int, device, dtype) -> torch.Tensor:
        # upper-triangular mask (True on masked positions)
        return torch.ones(q_len, k_len, device=device, dtype=torch.bool).triu(diagonal=1)

    def forward(
        self,
        query_states: torch.Tensor,                 # (B, Tq, D)
        kv_states: torch.Tensor,                    # (B, Tk, D)
        attention_mask: Optional[torch.Tensor] = None,   # (B, 1, Tq, Tk) additive mask (0 or -inf)
        kv_padding_mask: Optional[torch.Tensor] = None,   # (B, Tk) True for PAD
        attn_bias: Optional[torch.Tensor] = None,         # (B or 1, H or 1, Tq, Tk) if any
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (past_k, past_v), each (B,H,Tpast,Dh)
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        B, Tq, D = query_states.shape
        _, Tk, _ = kv_states.shape
        dtype = query_states.dtype
        device = query_states.device

        # Projections
        q = self.q_proj(query_states).to(dtype)
        k = self.k_proj(kv_states).to(dtype)
        v = self.v_proj(kv_states).to(dtype)

        # Split heads
        q = self._split_heads(q)   # (B,H,Tq,Dh)
        k = self._split_heads(k)   # (B,H,Tk,Dh)
        v = self._split_heads(v)   # (B,H,Tk,Dh)

        # Cache concat (옵션)
        present = None
        if layer_past is not None:
            past_k, past_v = layer_past
            # 과거 k/v 뒤에 현 k/v 이어붙임
            k = torch.cat([past_k.to(dtype), k], dim=-2)   # seq dim
            v = torch.cat([past_v.to(dtype), v], dim=-2)
        if use_cache:
            present = (k, v)

        # Scaled dot-product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,Tq,Tk_total)

        # Optional biases (e.g., ALiBi)
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias

        # Causal mask (필요 시)
        if self.is_causal:
            q_len, k_len = attn_scores.shape[-2], attn_scores.shape[-1]
            causal = self._causal_mask(q_len, k_len, device=device, dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(causal, float("-inf"))

        # Padding mask on K/V
        if kv_padding_mask is not None:
            # (B,1,1,Tk_total)
            mask = kv_padding_mask[:, None, None, :].to(torch.bool)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        # Additive attention mask (HF 스타일: (B,1,Tq,Tk_total), 0/-inf)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        ctx = torch.matmul(attn_probs, v)           # (B,H,Tq,Dh)
        ctx = self._merge_heads(ctx)                # (B,Tq,D)
        ctx = self.o_proj(ctx)

        if self.gated_residual:
            g = torch.sigmoid(self.gate).to(dtype)
            ctx = ctx * g

        out = query_states + self.resid_drop(ctx)

        # if self.gamma is not None:
        #     out = out * self.gamma

        return out, present


class scriptInjectionBlock(nn.Module):
    """
    범용 script cross-attn 주입 래퍼.
    기존 커스텀 블록들(GPT2Block/Qwen2DecoderLayer 등)처럼
    - 입력을 (hidden_states, before_pooled_states) 튜플 형태로 받아서
    - cross-attn을 수행하고
    - 텐서 하나(B, T, D)만 반환.
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        use_bias: bool = True,
        gated_residual: bool = True,
        is_causal: bool = True,
        layer_scale_init: float = 0.0,
    ):
        super().__init__()
        self.cross_attn = GenericCrossAttention(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            use_bias=use_bias,
            gated_residual=gated_residual,
            is_causal=is_causal,
            layer_scale_init=layer_scale_init,
        )

    def forward(
        self,
        hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Sequence[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,       # 호환용 인자(미사용)
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 호환용(미사용)
        encoder_attention_mask: Optional[torch.Tensor] = None, # 호환용(미사용)
        position_ids: Optional[torch.LongTensor] = None,       # 호환용(미사용)
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # 1) 입력 언팩: (query, kv) or 별도 키워드
        if isinstance(hidden_states, (tuple, list)) and len(hidden_states) == 2:
            query_states, kv_states = hidden_states
        else:
            query_states = kwargs.get("query_states", None)
            kv_states = kwargs.get("kv_states", None)
            if query_states is None or kv_states is None:
                raise ValueError(
                    "scriptInjectionBlock.forward expects hidden_states=(query, kv) "
                    "or explicit query_states=..., kv_states=..."
                )

        # 2) padding 마스크(옵션): kv_padding_mask 우선순위
        kv_padding_mask = kwargs.get("kv_padding_mask", None)

        # 3) cross-attn
        fused, _present = self.cross_attn(
            query_states=query_states,
            kv_states=kv_states,
            attention_mask=attention_mask,
            kv_padding_mask=kv_padding_mask,
            attn_bias=kwargs.get("attn_bias", None),
            layer_past=layer_past,
            use_cache=use_cache,
        )

        # 4) 출력 일원화: 텐서 하나만 반환(네 기존 커스텀 블록과 동일)
        return fused
