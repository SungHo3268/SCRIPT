import os
import sys
from sys import exit
import textwrap
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence as _pad
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Config, GPTJConfig, GPTNeoXConfig, BertConfig, Qwen2Config
sys.path.append(os.getcwd())
from srcs.functions import trim_pad, repeat_interleave
from srcs.attn_pool import CustomGPT2Block, CustomBertAttention
from srcs.gptj_attn_pool import CustomGPTJBlock
from srcs.gptneox_attn_pool import CustomGPTNeoXLayer
from srcs.qwen_attn_pool import CustomQwen2DecoderLayer
from srcs.lora import LoRA_Config, LoRA_Layer, apply_lora_to_model
from srcs.generic_script_injection import scriptInjectionBlock


class LinearnAddnNorm(nn.Module):
    def __init__(self, max_length, d_model):
        super(LinearnAddnNorm, self).__init__()
        self.sublayer = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Linear(max_length*2, max_length),
            Rearrange('b d n -> b n d'),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        residual = x1
        x = self.sublayer(torch.cat([x1, x2], dim=1))
        # x = self.norm(x + residual)
        x += residual
        return x


class Pooling(nn.Module):
    def __init__(self, pooling_module, pooling_size):
        super().__init__()
        self.pooling_module = pooling_module
        self.pooling_size = pooling_size

    def forward(self, x):
        before_pooled_states = rearrange(x, 'b n (k d) -> b (n k) d', k=self.pooling_size)
        after_pooled_states = self.pooling_module(x)
        return [after_pooled_states, before_pooled_states]


class script_Config:
    def __init__(self, tok_type, reducer, hidden_dim, script_max_length, max_length,
                 do_combination, combination_type, trans_config, num_attention_heads, intermediate_size, num_trans_layers,
                 dropout=0.0,
                 is_bert=False, fusion="cross_attn", lora_config: LoRA_Config=None):
        self.tok_type = tok_type
        self.reducer = reducer
        self.hidden_dim = hidden_dim
        self.script_max_length = script_max_length
        self.max_length = max_length
        try:
            assert script_max_length % max_length == 0       # Ensure the script_max_length is divisible by max
        except AssertionError:
            print("\n\n\n")
            print(f"* script_max_length: {script_max_length}")
            print(f"* max_length: {max_length}")
            print(f"* script_max_length must be divisible by max_length.")
            print("\n\n\n")
        self.k = script_max_length // max_length

        self.do_combination = do_combination
        self.combination_type = combination_type
        self.trans_config = trans_config
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_trans_layers = num_trans_layers
        self.dropout = dropout

        self.is_bert = is_bert
        self.fusion = fusion
        self.lora_config = lora_config

class script_Combination_Layer(nn.Module):
    def __init__(self, config: script_Config, script_tokenizer, original_tokenizer):
        super(script_Combination_Layer, self).__init__()
        self.config = config
        self.original_tokenizer = original_tokenizer

        self.script_tokenizer = script_tokenizer
        self.pad_token_id = script_tokenizer.pad_token_id
        self.eos_token_id = script_tokenizer.eos_token_id
        if config.is_bert:
            self.unk_token_id = script_tokenizer.unk_token_id
            self.sep_token_id = script_tokenizer.sep_token_id
            self.mask_token_id = script_tokenizer.mask_token_id

        if config.tok_type == 'stroke_var':
            self.cho_len = 4
            self.joong_len = 1
            self.jong_len = 4
            self.max_jamo_len = max(self.cho_len, self.joong_len, self.jong_len)
            self.char_len = self.cho_len + self.joong_len + self.jong_len
        elif config.tok_type == 'cji_var':
            self.cho_len = 1
            self.joong_len = 5
            self.jong_len = 1
            self.max_jamo_len = max(self.cho_len, self.joong_len, self.jong_len)
            self.char_len = self.cho_len + self.joong_len + self.jong_len
        elif config.tok_type == 'bts_var':
            self.cho_len = 4
            self.joong_len = 5
            self.jong_len = 4
            self.max_jamo_len = max(self.cho_len, self.joong_len, self.jong_len)
            self.char_len = self.cho_len + self.joong_len + self.jong_len
        elif config.tok_type in ['jamo_var', 'jamo_distinct']:
            self.cho_len = 1
            self.joong_len = 1
            self.jong_len = 1
            self.max_jamo_len = max(self.cho_len, self.joong_len, self.jong_len)
            self.char_len = self.cho_len + self.joong_len + self.jong_len
        else:
            pass

        self.script_embedding = nn.Embedding(num_embeddings=len(script_tokenizer),
                                            embedding_dim=config.hidden_dim)

        if config.do_combination:
            if config.combination_type == 'gru':
                self.contextualization = nn.Sequential(
                    nn.GRU(
                        input_size=config.hidden_dim,
                        hidden_size=config.hidden_dim,
                        num_layers=1,
                        batch_first=True,
                    )
                )
            else:
                raise NotImplementedError

            self.add_jongsung = nn.Sequential(
                Rearrange('b l n d -> b d l n'),
                nn.Conv2d(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim,
                    kernel_size=(2, 1),
                    stride=1,
                    padding='same',
                    padding_mode='zeros',
                    groups=config.hidden_dim
                ),
                nn.AvgPool2d(
                    kernel_size=(2, 1),
                    stride=1,
                    count_include_pad=True,
                ),
                Rearrange('b d l n -> b l n d')
            )
            self.get_org_shape_emb = nn.GRU(
                    input_size=config.hidden_dim,
                    hidden_size=config.hidden_dim,
                    num_layers=1,
                    batch_first=True,
                )

            self.init_combination_layer()
        else:
            if config.reducer == 'linear':
                self.sequence_reducer = nn.Sequential(
                    Rearrange('b s d -> b d s'),
                    nn.Linear(config.script_max_length, config.max_length),
                    Rearrange('b d s -> b s d'),
                    nn.LayerNorm(config.hidden_dim)
                )
            elif config.reducer == 'linear_pool':           # Hourglass Transformer
                self.sequence_reducer = nn.Sequential(
                    Rearrange('b (n k) d -> b n k d', k=config.k),
                    Rearrange('b n k d -> b n (k d)'),
                    nn.Linear(config.k * config.hidden_dim, config.hidden_dim),
                    # nn.LayerNorm(config.hidden_dim),
                )
            elif config.reducer == 'attention_pool':        # Funnel Transformer
                attn_pool_config = config.trans_config
                attn_pool_config.update({'is_cross_attention': True})
                print("\n\n")
                print(f"attn_pool_config: {attn_pool_config}")

                self.sequence_reducer = nn.Sequential(
                    Rearrange('b (n k) d -> b n k d', k=config.k),
                    Rearrange('b n k d -> b n (k d)'),
                    Pooling(nn.AvgPool1d(kernel_size=config.k,
                                         stride=config.k,
                                         count_include_pad=True
                                         ),
                            config.k
                            ),
                    CustomGPT2Block(attn_pool_config, layer_idx=0),
                )
            else:
                raise NotImplementedError

            self.init_reducer()
            self.script_embedding.weight.data.normal_(mean=0.0, std=0.02)

    def init_combination_layer(self):
        print("Init combination layer")
        if self.config.combination_type == 'gru':
            self.contextualization[0].weight_hh_l0.data.normal_(mean=0.0, std=0.02)
            self.contextualization[0].weight_ih_l0.data.normal_(mean=0.0, std=0.02)
            self.contextualization[0].bias_hh_l0.data.zero_()
            self.contextualization[0].bias_ih_l0.data.zero_()
        else:
            raise NotImplementedError
        self.add_jongsung[1].weight.data.normal_(mean=0.0, std=0.02)
        self.add_jongsung[1].bias.data.zero_()
        self.get_org_shape_emb.weight_hh_l0.data.normal_(mean=0.0, std=0.02)
        self.get_org_shape_emb.weight_ih_l0.data.normal_(mean=0.0, std=0.02)
        self.get_org_shape_emb.bias_hh_l0.data.zero_()
        self.get_org_shape_emb.bias_ih_l0.data.zero_()

    def init_reducer(self):
        print("Init reducer")
        if self.config.reducer == 'linear':
            self.sequence_reducer[1].weight.data.normal_(mean=0.0, std=0.02)
            self.sequence_reducer[1].bias.data.zero_()
        elif self.config.reducer == 'linear_pool':
            self.sequence_reducer[2].weight.data.normal_(mean=0.0, std=0.02)
            self.sequence_reducer[2].bias.data.zero_()
        elif self.config.reducer == 'attention_pool':
            pass
        else:
            raise NotImplementedError

    @torch._dynamo.disable()
    def flatten_rnns(self):
        if hasattr(self, "contextualization"):
            for m in self.contextualization.modules():
                if isinstance(m, nn.GRU):
                    m.flatten_parameters()

        if hasattr(self, "get_org_shape_emb") and isinstance(self.get_org_shape_emb, nn.GRU):
            self.get_org_shape_emb.flatten_parameters()

    @torch._dynamo.disable()
    def _run_gru(self, gru_mod: nn.GRU, x: torch.Tensor):
        # 연속 메모리화(성능 최적화) - 그래프 밖에서 실행
        try:
            gru_mod.flatten_parameters()
        except Exception:
            pass
        return gru_mod(x)  # (out, h_n)

    @torch._dynamo.disable()
    def _align_and_pad_to_subwords(self, script_embedding, text_input, char_seq_len, device):
        """
        script_embedding: Tensor (B, N_char, D)
        text_input: List[str]
        char_seq_len: int
        return: Tensor (B, N_subword_padded, D) with right padding to max_length (or max_length-1 for BERT)
        """
        end_indices = []
        for text in text_input:
            if 'mGPT' in self.original_tokenizer.name_or_path:
                tokens, continuous = [], []
                for tok in self.original_tokenizer.encode(text):
                    if len(continuous) > 0:
                        continuous.append(tok)
                        decoded = self.original_tokenizer.decode(continuous)
                    else:
                        decoded = self.original_tokenizer.decode(tok)
                    if '�' in decoded:
                        if len(continuous) == 0:
                            continuous.append(tok)
                        continue
                    else:
                        continuous = []
                        tokens.append(decoded)
            else:
                tokens = self.original_tokenizer.tokenize(text)

            if len(tokens) != 0:
                if (len(tokens) > 1) and (tokens[-1] == "▁"):
                    tokens = tokens[:-1]
                tokens[0] = tokens[0].replace("▁", "")
                if len(tokens[0]) == 0:
                    tokens = tokens[1:]

            end_idx = np.cumsum([len(w) for w in tokens]) - 1
            end_idx = [i for i in end_idx if i < char_seq_len]
            end_indices.append(end_idx)

        B = script_embedding.shape[0]
        picked = [script_embedding[i, end_indices[i]] for i in range(B)]  # list of Tensors
        script_sw = _pad(picked, batch_first=True, padding_value=self.pad_token_id).to(device)  # (B, N_subword, D)

        return end_indices, script_sw

    def forward(self, x, text_input):
        """
        :param x: input for script layer which size of (batch_size, jamo_seq_len(=script_max_length)), this is the input_ids of jamo.
        """

        script_embedding = self.script_embedding(x)       # (batch_size, jamo_seq_len(=script_max_length), hidden_dim) = (B, N_max_jamo, D)
        # print("\n")
        # print(f"1) script_embedding.shape: {script_embedding.shape}")
        if self.config.do_combination:
            batch_size = script_embedding.shape[0]
            if self.config.is_bert:
                x = x[:, 1:]                                    # Remove the [CLS] token
                script_cls_embedding = script_embedding[:, :1, :]
                script_embedding = script_embedding[:, 1:, :]       # Remove the [CLS] token
                # print(f"1.1) script_embedding.shape: {script_embedding.shape}")

                sep_token_idx = (x == self.sep_token_id) + (x == self.unk_token_id) + (x == self.mask_token_id)
                repeat_num = torch.ones_like(x).to(x.device)
                repeat_num[sep_token_idx] = self.char_len

                # Calculate the context_input_ids too.
                x = repeat_interleave(x, repeats=repeat_num.detach().cpu(), dim=1)
                script_embedding = repeat_interleave(script_embedding, repeats=repeat_num.detach().cpu(), dim=1)
                x, script_embedding = trim_pad(x, script_embedding, pad_value=self.pad_token_id)

            x, script_embedding = trim_pad(x, script_embedding, pad_value=self.pad_token_id)      # (B, N_jamo, D)
            # print(f"2) script_embedding.shape: {script_embedding.shape}")

            try:
                assert x.shape[1] % self.char_len == 0
            except AssertionError:
                print("\n\n\n")
                print(text_input)
                print(x.shape)
                print(x)
                exit(-111)

            assert x.shape[1] == script_embedding.shape[1]

            # Get character representative input_ids
            x = x[:, ::self.char_len]

            # Get character representative embeddings
            """
            1) Contextualization
            """
            # script_embedding = self.contextualization(script_embedding)[0]        # (B, N_jamo, D)
            gru = self.contextualization[0]
            script_embedding = self._run_gru(gru, script_embedding)[0]
            # print(f"3) script_embedding.shape: {script_embedding.shape}")
            script_embedding = script_embedding.reshape(batch_size, -1, self.char_len, self.config.hidden_dim)        # # (B, N_jamo, D) -> (B, N_char, 3, D)
            # print(f"4) script_embedding.shape: {script_embedding.shape}")
            """
            2) Fusion of Chosung and Joongsung
            """
            cho_joong_inputs = torch.mean(script_embedding[:, :, :- self.jong_len], dim=2, keepdim=True)      # (B, N_char, 2, D)
            # print(f"5) cho_joong_inputs.shape: {cho_joong_inputs.shape}")
            jong_inputs = torch.mean(script_embedding[:, :, -self.jong_len:], dim=2, keepdim=True)        # (B, N_char, 1, D)
            # print(f"6) jong_inputs.shape: {jong_inputs.shape}")
            script_embedding = torch.concat([cho_joong_inputs, jong_inputs], dim=2)               # (B, N_char, 2, D)
            # print(f"7) script_embedding.shape: {script_embedding.shape}")             # (B, N_char, 2, D)

            """
            3) Addition of Jongsung (Rearrange & Conv)
            """
            script_embedding = rearrange(script_embedding, 'b n l d -> b l n d')      # (B, N_char, 2, D) -> (B, 2, N_char, D)
            script_embedding = self.add_jongsung(script_embedding).squeeze()          # (B, 2, N_char, D) -> (B, N_char, D)

            # print(f"8) script_embedding.shape: {script_embedding.shape}")
            if script_embedding.ndim == 2:           # it runs when the batch size is 1.
                if script_embedding.shape[0] == batch_size:
                    script_embedding = script_embedding.unsqueeze(1)        # seq_num=1
                else:
                    script_embedding = script_embedding.unsqueeze(0)        # batch_size=1
            elif script_embedding.ndim == 1:
                script_embedding = script_embedding.reshape(1, 1, -1)       # batch_size=1, seq_num=1

            # '''
            """
            4) Get original token representative embeddings (e.g., subword, morpheme, etc.)
            """
            script_embedding = script_embedding.to(self.get_org_shape_emb.weight_hh_l0.dtype)

            # script_embedding = self.get_org_shape_emb(script_embedding)[0]        # (B, N_char, D)
            script_embedding = self._run_gru(self.get_org_shape_emb, script_embedding)[0]

            char_seq_len = script_embedding.shape[1]


            # print(f"9) script_embedding.shape: {script_embedding.shape}")
            end_indices, script_embedding = self._align_and_pad_to_subwords(
                script_embedding, text_input, char_seq_len, x.device
            )

            try:
                if self.config.is_bert:
                    if script_embedding.shape[1] > self.config.max_length - 1:
                        script_embedding = script_embedding[:, :self.config.max_length - 1]
                    pad_len = self.config.max_length - 1 - script_embedding.shape[1]
                    if pad_len > 0:
                        pad = torch.full(
                            (script_embedding.size(0), pad_len, self.config.hidden_dim),
                            fill_value=self.pad_token_id, device=x.device, dtype=script_embedding.dtype
                        )
                        script_embedding = torch.cat([script_embedding, pad], dim=1)
                else:
                    pad_len = self.config.max_length - script_embedding.shape[1]
                    if pad_len > 0:
                        pad = torch.full(
                            (script_embedding.size(0), pad_len, self.config.hidden_dim),
                            fill_value=self.pad_token_id, device=x.device, dtype=script_embedding.dtype
                        )
                        script_embedding = torch.cat([script_embedding, pad], dim=1)
            except Exception:
                for i, t in enumerate(text_input):
                    print(f"text_input[{i}]: {t}")
                    # print(f"end_indices[{i}]: {end_indices[i]}")
                exit(-111)
            # print(f"11) script_embedding.shape: {script_embedding.shape}")
            # '''
            """

            end_indices = []
            for text in text_input:
                if 'mGPT' in self.original_tokenizer.name_or_path:
                    tokens = []
                    continuous_token = []
                    for tok in self.original_tokenizer.encode(text):
                        if len(continuous_token) > 0:
                            continuous_token.append(tok)
                            decoded = self.original_tokenizer.decode(continuous_token)
                        else:
                            decoded = self.original_tokenizer.decode(tok)

                        if '�' in decoded:
                            if len(continuous_token) > 0:
                                pass
                            else:
                                continuous_token.append(tok)
                            continue
                        else:
                            continuous_token = []
                            tokens.append(decoded)
                else:
                    tokens = self.original_tokenizer.tokenize(text)

                if len(tokens) != 0:
                    if (len(tokens) > 1) and (tokens[-1] == "▁"):
                        tokens = tokens[:-1]

                    tokens[0] = tokens[0].replace("▁", "")
                    if len(tokens[0]) == 0:
                        tokens = tokens[1:]

                end_idx = np.cumsum([len(word) for word in tokens]) - 1  # '-1' is for making index start from 0.
                end_idx = [idx for idx in end_idx if idx < char_seq_len]  # Remove the index which is out of the range.

                end_indices.append(end_idx)

            script_embedding = [script_embedding[i, end_indices[i]] for i in range(batch_size)]  # (B, N_char, D) -> (B, N_subword(not consistent), D)
            script_embedding = pad_sequence(script_embedding, batch_first=True, padding_value=self.pad_token_id)  # (B, N_subword, D)
            # print(f"10) script_embedding.shape: {script_embedding.shape}")
            
            
            # Padding
            try:
                if self.config.is_bert:
                    if script_embedding.shape[1] > self.config.max_length - 1:
                        print(f"There may be wrong length of script_embedding sentences.")
                        script_embedding = script_embedding[:, :self.config.max_length - 1]

                    script_embedding = torch.concat([
                        script_embedding,
                        torch.full(
                            size=(batch_size, self.config.max_length - 1 - script_embedding.shape[1],
                                  self.config.hidden_dim),  # exclude the [CLS] token
                            fill_value=self.pad_token_id, device=x.device)
                    ], dim=1)  # (B, N_subword, D) -> (B, max_length, D)
                else:
                    script_embedding = torch.concat([
                        script_embedding,
                        torch.full(
                            size=(batch_size, self.config.max_length - script_embedding.shape[1], self.config.hidden_dim),
                            fill_value=self.pad_token_id, device=x.device)
                    ], dim=1)  # (B, N_subword, D) -> (B, max_length, D)
            except:
                for i in range(batch_size):
                    print(f"text_input[{i}]: {text_input[i]}")
                    print(f"end_indices[{i}]: {end_indices[i]}")
                exit(-111)
            """

        else:
            if self.config.is_bert:
                script_cls_embedding = script_embedding[:, :1, :]
                script_embedding = script_embedding[:, 1:, :]       # Remove the [CLS] token
            script_embedding = self.sequence_reducer(script_embedding)        # (B, max_length, D)

        if self.config.is_bert:
            script_embedding = torch.cat([script_cls_embedding, script_embedding], dim=1)
        return script_embedding

class script_LoRA_Layer(nn.Module):
    """
    Be careful with the name of weight parameters. Only the weight of script layer should have the name 'script_'.
    The name which has 'script_' will be trained. The other weights will be frozen.
    """
    def __init__(self, tokenizer, script_tokenizer, original_layer, config: script_Config):
        super(script_LoRA_Layer, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.script_tokenizer = script_tokenizer

        self.original_layer = original_layer
        if config.hidden_dim is None:
            config.hidden_dim = original_layer.weight.shape[1]

        self.script_combination = script_Combination_Layer(config, script_tokenizer, tokenizer)

        # Use each model’s attention block to set the cross-attention block.
        # The core customized implementations are as follows:
        # 1) Use the "original representation for the Query",
        # 2) Employ the "script-compressed representation for the Key and Value".
        # 3) Activate only the direct functions for Q, K, and V in each method (e.g., AttentionBlock)
        # 4) Deactivate other techniques such as positional embeddings.
        # 5) Remove any normalization function from the representation, since the two heterogeneous representations largely differ in size.
        # 6) Output of "cross_attention_block(=script_injection)" is solely the "hidden_states"
        # However, it would be better to integrate all models into a single cross-attention block. (FUTURE WORK)
        if config.fusion == 'cross_attn':
            self.script_injection = scriptInjectionBlock(
                hidden_size=config.hidden_dim,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_attention_heads,
                attn_dropout=config.dropout,
                resid_dropout=config.dropout,
                use_bias=False,
                gated_residual=True,
                is_causal=True,
                layer_scale_init=0.0,
            )

            # if isinstance(config.trans_config, GPT2Config):
            #     self.script_injection = CustomGPT2Block(config.trans_config, layer_idx=0)
            # elif isinstance(config.trans_config, Qwen2Config):
            #     self.script_injection = CustomQwen2DecoderLayer(config.trans_config, layer_idx=0)
            # elif isinstance(config.trans_config, GPTJConfig):
            #     self.script_injection = CustomGPTJBlock(config.trans_config)
            # elif isinstance(config.trans_config, GPTNeoXConfig):
            #     self.script_injection = CustomGPTNeoXLayer(config.trans_config)
            # elif isinstance(config.trans_config, BertConfig):
            #     self.script_injection = CustomBertAttention(config.trans_config)
        elif config.fusion == 'concat':
            self.script_concat = LinearnAddnNorm(config.max_length, config.hidden_dim)

        injection_config = config.trans_config
        injection_config.update({'is_cross_attention': False})

        base_w = original_layer.weight
        self.script_combination.to(device=base_w.device, dtype=base_w.dtype)
        if hasattr(self, "script_concat"):
            self.script_concat.to(device=base_w.device, dtype=base_w.dtype)

        self.script_combination.flatten_rnns()


    @property
    def weight(self):
        return self.original_layer.weight

    @weight.setter
    def weight(self, val):
        self.original_layer.weight = val

    @property
    def num_embeddings(self):
        w = self.original_layer.weight
        return w.size(0)

    @property
    def embedding_dim(self):
        w = self.original_layer.weight
        return w.size(1)

    def make_script_input(self, x, device):
        """
        :param x: original input text which is the string type.
        :return: script_x: input for script layer which size of (batch_size, jamo_seq_len)
        """
        script_encoded = self.script_tokenizer(
            x, padding="max_length", truncation=True,
            max_length=self.config.script_max_length, return_tensors='pt')
        script_x = script_encoded['input_ids'].to(device)
        return script_x

    @torch._dynamo.disable()  # Outside of the dynamo graph
    def _decode_and_make_script_ids(self, x, device):
        if self.config.is_bert:
            text_input = self.tokenizer.batch_decode(x, skip_special_tokens=False)
            text_input = [text.replace("[CLS]", "").replace("[PAD]", "").strip() for text in text_input]
        else:
            text_input = self.tokenizer.batch_decode(x, skip_special_tokens=True)

        script_x = self.make_script_input(text_input, device)                 # (B, N(=max_script_length), D)
        return text_input, script_x

    def forward(self, x):
        #############################################################################################
        #    the input of the script_LoRA Layer is the tokenized sequence by original PLM's tokenizer
        #############################################################################################
        device = x.device

        original_embedding = self.original_layer(x)

        # if self.config.is_bert:
        #     text_input = self.tokenizer.batch_decode(x, skip_special_tokens=False)
        #     text_input = [text.replace("[CLS]", "").replace("[PAD]", "").strip() for text in text_input]
        #     # print("\n")
        #     # print(f"text_input[0]: {text_input[0]}")
        # else:
        #     text_input = self.tokenizer.batch_decode(x, skip_special_tokens=True)
        #
        # script_x = self.make_script_input(text_input, device)             # (B, N(=max_script_length), D)

        text_input, script_x = self._decode_and_make_script_ids(x, device)

        script_embedding = self.script_combination(script_x, text_input)       # (B, N(=max_length), D)

        if original_embedding.shape[1] != script_embedding.shape[1]:
            script_embedding = script_embedding[:, :original_embedding.shape[1]]

        script_embedding = script_embedding.to(dtype=original_embedding.dtype, device=original_embedding.device)
        if self.config.fusion == 'sum' or self.config.is_bert:
            final_embedding = original_embedding + script_embedding
        elif self.config.fusion == 'concat':
            final_embedding = self.script_concat(original_embedding, script_embedding)
        elif self.config.fusion == 'cross_attn':
            out = self.script_injection([original_embedding, script_embedding.contiguous()])
            if isinstance(out, tuple):
                final_embedding = out[0]
            elif hasattr(out, "last_hidden_state"):
                final_embedding = out.last_hidden_state
            else:
                final_embedding = out
        else:
            raise NotImplementedError

        if not torch.is_tensor(final_embedding):
            raise TypeError(f"script_LoRA_Layer.forward must return Tensor, got {type(final_embedding)}")
        assert original_embedding.shape == final_embedding.shape, \
            f"Shape mismatch: orig={tuple(original_embedding.shape)} vs final={tuple(final_embedding.shape)}"

        return final_embedding

    def __repr__(self):
        def _shape_of_script_embedding(script_emb):
            if isinstance(script_emb, nn.Embedding):
                try:
                    return tuple(script_emb.weight.shape)
                except Exception:
                    return (getattr(script_emb, "num_embeddings", "<?>"),
                            getattr(script_emb, "embedding_dim", "<?>"))
            if hasattr(script_emb, "shape"):
                return tuple(script_emb.shape)
            if hasattr(script_emb, "size"):
                try:
                    return tuple(script_emb.size())
                except Exception:
                    pass
            ne = getattr(script_emb, "num_embeddings", None)
            ed = getattr(script_emb, "embedding_dim", None)
            if ne is not None and ed is not None:
                return (ne, ed)
            return ("<?>", "<?>")

        def _indent(s, n=2):
            # 문자열 s를 n칸 들여쓰기
            return textwrap.indent(s, " " * n)

        def _label_and_block(label: str, obj: object, base=2, extra=4):
            """
            '  (label):\n' + obj의 repr을 base+extra 들여쓰기로 붙여주는 헬퍼.
            obj가 None이면 생략.
            """
            if obj is None:
                return ""
            block = repr(obj)
            return _indent(f"({label}):\n", base) + _indent(block + "\n", base + extra)

        script_shape = _shape_of_script_embedding(self.script_combination.script_embedding)

        lines = []
        lines.append(f"{self.__class__.__name__}(")

        # original_layer 한 줄
        lines.append(_indent(f"(original_layer): {repr(self.original_layer)},\n", 2))

        # script_embedding 한 줄
        # (Embedding/Parameter 타입에 상관없이 shape만 명시)
        lines.append(_indent(f"(script_embedding): Embedding{script_shape},\n", 2))

        if self.config.do_combination:
            # script_Combination_Layer 묶어서 블록 표현
            lines.append(_indent(f"(script_combination): script_Combination_Layer(\n", 2))

            # contextualization/sequence_reducer가 존재하면 블록으로 표현
            if hasattr(self.script_combination, "contextualization"):
                lines.append(_label_and_block("contextualization",
                                              self.script_combination.contextualization,
                                              base=4, extra=2))
            if hasattr(self.script_combination, "sequence_reducer"):
                lines.append(_label_and_block("sequence_reducer",
                                              self.script_combination.sequence_reducer,
                                              base=4, extra=2))

            # 닫기
            lines.append(_indent(")\n", 2))
        else:
            # do_combination=False일 때는 sequence_reducer만 블록으로
            if hasattr(self.script_combination, "sequence_reducer"):
                lines.append(_label_and_block("sequence_reducer",
                                              self.script_combination.sequence_reducer,
                                              base=2, extra=2))

        lines.append(")")
        return "".join(lines)


# def apply_script_to_model(model, tokenizer, script_tokenizer, script_config: script_Config, logger=None):
#     print("\n")
#     for name, module in model.named_modules():
#         hierarchy = name.split('.')
#         layer_name = hierarchy[-1]
#         if len(hierarchy) > 1 and layer_name in ['wte', 'embed_in', 'word_embeddings', 'embed_tokens']:    # Ensure the module is not the top-level module
#             parent_module = model
#
#             parent_names = hierarchy[:-1]
#             for submodule_name in parent_names:  # Navigate to the parent module
#                 parent_module = getattr(parent_module, submodule_name)
#
#             original_layer = getattr(parent_module, layer_name)
#             if isinstance(original_layer, nn.Embedding):
#                 script_layer = script_LoRA_Layer(tokenizer, script_tokenizer, original_layer, script_config)
#                 setattr(parent_module, layer_name, script_layer)
#                 if logger:
#                     logger.info(f"Replaced {name} with script_LoRA_Layer")
#                 else:
#                     print(f"Replaced [ ' {name} ' ] layer with [ ' script_LoRA_Layer ' ]")
#     return model

def _retie_weights_if_needed(model):
    if getattr(getattr(model, "config", None), "tie_word_embeddings", True):
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        elif hasattr(model, "_tie_or_clone_weights"):
            model._tie_or_clone_weights(None)

def apply_script_to_model(model, tokenizer, script_tokenizer, script_config: script_Config, logger=None):
    # 1) Original Embedding
    if hasattr(model, "bert"):
        original_emb = model.bert.get_input_embeddings()
    else:
        original_emb = model.get_input_embeddings()
    assert isinstance(original_emb, nn.Embedding), "expected nn.Embedding as base input embeddings"

    # 2) Make Embedding Layer with script
    script_layer = script_LoRA_Layer(tokenizer, script_tokenizer, original_emb, script_config)

    # 3) Replacing
    if hasattr(model, "bert"):
        model.bert.set_input_embeddings(script_layer)
    else:
        model.set_input_embeddings(script_layer)
    if logger: logger.info("Replaced input embeddings with script_LoRA_Layer.")
    else: print("Replaced input embeddings with script_LoRA_Layer.")

    # 4) Restore tie weights
    _retie_weights_if_needed(model)

    # 5) shape sanity check
    if hasattr(model, "bert"):
        emb = model.bert.get_input_embeddings()
    else:
        emb = model.get_input_embeddings()
    lm_head = getattr(model, "lm_head", None)
    if hasattr(emb, "weight") and lm_head is not None and hasattr(lm_head, "weight"):
        if emb.weight.shape[1] != lm_head.weight.shape[1]:
            print(f"[WARN] hidden dim mismatch: embed{tuple(emb.weight.shape)} vs lm_head{tuple(lm_head.weight.shape)}")

    return model


def make_only_script_and_lora_as_trainable(model, weight: str = 'none', bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if weight == 'script_lora_only':
            p.requires_grad = ('script_' in n) or ('lora_' in n)
        elif weight == 'script_only':
            p.requires_grad = ('script_' in n)
        elif weight == 'lora_only':
            p.requires_grad = ('lora_' in n)
        else:
            p.requires_grad = False

    if bias == 'none':
        pass
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'script_only':
        for m in model.modules():
            if isinstance(m, script_Combination_Layer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, script_LoRA_Layer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
    elif bias == 'script_lora_only':
        for m in model.modules():
            if ((isinstance(m, script_Combination_Layer) or isinstance(m, LoRA_Layer))
                    and hasattr(m, 'bias') and m.bias is not None):
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


if __name__ == "__main__":
    from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel
    import os
    import sys
    sys.path.append(os.getcwd())
    from srcs.lora import print_trainable_parameters
    from pretraining.scripts.run_gpt_pretraining import get_gpt2_tokenizer

    model_name = "skt/kogpt2-base-v2"

    tok_type = "jamo_var"
    script_max_length = 2048
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    script_tokenizer = get_gpt2_tokenizer(tok_type=tok_type, lang="ko", max_length=script_max_length,
                                         lowercase=True, clean_text=True, add_bos_token=False,
                                         bos_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>", unk_token="<unk>")


    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Check the original number of parameters
    origin_num = sum(p.numel() for p in model.parameters())
    print("Original number of parameters:", origin_num)

    # Configuration for Transformer
    trans_config = AutoConfig.from_pretrained(model_name)

    # Configuration for LoRA
    lora_config = LoRA_Config(
        r=16,
        lora_alpha=64,
        lora_dropout=0.03,
        target_modules=["c_attn", "c_proj"],
    )
    script_config = script_Config(
        tok_type=tok_type,
        reducer="linear",
        hidden_dim=768,
        script_max_length=script_max_length,
        max_length=256,
        do_combination=False,
        combination_type='gru',
        trans_config=trans_config,
        num_attention_heads=3,
        intermediate_size=3072,
        num_trans_layers=3,
        is_bert=False,
        fusion="cross_attn",
        lora_config=lora_config
    )

    # Apply LoRA to the model
    model = apply_script_to_model(model, tokenizer, script_tokenizer, script_config)
    model = apply_lora_to_model(model, lora_config)
    make_only_script_and_lora_as_trainable(model, weight='script_lora_only', bias='script_lora_only')
    _, _ = print_trainable_parameters(model)