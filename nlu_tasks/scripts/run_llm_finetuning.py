import re
import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from distutils.util import strtobool as _bool

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cpu\.amp\.autocast.*"
)

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 128
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
# torch._dynamo.config.accumulated_cache_size_limit = 512

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import evaluate
import importlib
import numpy as np
from scipy.stats import pearsonr, spearmanr
from datasets import Dataset
from dataclasses import dataclass
from typing import List, Dict, Any
from peft import LoraConfig, get_peft_model, TaskType, PeftConfig
from transformers import (AutoTokenizer, AutoConfig, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoModelForMultipleChoice,
                          DataCollatorWithPadding, DataCollatorForMultipleChoice,
                          TrainingArguments, Trainer,
                          EvalPrediction, EarlyStoppingCallback)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

sys.path.append(os.getcwd())
from srcs.functions import init_random
from srcs.gpt_utils import text_tokenization_for_llm_classification, text_tokenization_for_llm_mc
from pretraining.scripts.run_gpt_pretraining import get_gpt2_tokenizer
from srcs.script import make_only_script_and_lora_as_trainable, apply_script_to_model, script_Config


def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, train


def print_trainable_parameters(model, prefix=None):
    tot0, tr0 = count_params(model)
    message = f"total={tot0:,}, trainable={tr0:,}"
    if prefix is not None:
        message = f"[{prefix}] {message}"
    print(message)


def get_nlu_dataset(args, tokenizer):
    task_util_path = f"nlu_tasks.data_utils.{args.task_name}.data_utils"
    task_util = importlib.import_module(task_util_path, package=".")
    if args.task_name in ['KorNLI', 'KorSTS', 'NSMC', 'PAWS_X']:
        dataset = task_util.load_task_dataset(args.remain_lang, args.do_hangeulize, args.data_remove)
    elif args.task_name in ['KB_BoolQ', 'KB_COPA', 'KB_WiC', 'KB_HellaSwag', 'KB_SentiNeg']:
        dataset = task_util.load_task_dataset()
    else:
        raise ValueError(f"It's a Wrong Task Name (entered '{args.task_name}'). Please enter the right task name among "
                          "[KorNLI, KorSTS, NSMC, PAWS_X] or "
                          "[KB_BoolQ, KB_COPA, KB_WiC, KB_HellaSwag, KB_SentiNeg]")

    total_dataset = {}
    for mode in ['train', 'dev', 'test']:
        data = Dataset.from_dict(dataset[mode])
        if args.task_name in ['KB_COPA', 'KB_HellaSwag']:      # For multiple choice tasks
            tokenized_datasets = data.map(text_tokenization_for_llm_mc,
                                          fn_kwargs={"tokenizer": tokenizer,
                                                     "max_length": args.max_input_length},
                                          remove_columns=data.column_names,
                                          batched=True,
                                          batch_size=args.batch_size // args.grad_acc,
                                          load_from_cache_file=False,
                                          desc="tokenize KB_COPA & KB_HellaSwag (causal-MC)",
                                          )
        else:       # For sentence classification tasks
            tokenized_datasets = data.map(text_tokenization_for_llm_classification,
                                          fn_kwargs={"tokenizer": tokenizer,
                                                     "max_length": args.max_input_length},
                                          remove_columns=data.column_names,
                                          batched=True,
                                          batch_size=args.batch_size // args.grad_acc,
                                          )
        total_dataset[mode] = tokenized_datasets
    return total_dataset


@dataclass
class DataCollatorForCausalMC:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # features: list of samples; each has input_ids: List[List[int]] shape (C, seq_i)
        # prompt_lens: List[int] len=C
        C = len(features[0]["input_ids"])
        # 길이 측정
        lens = []
        for f in features:
            for seq in f["input_ids"]:
                lens.append(len(seq))
        max_len = max(lens)

        if self.pad_to_multiple_of:
            remainder = max_len % self.pad_to_multiple_of
            if remainder != 0:
                max_len = max_len + (self.pad_to_multiple_of - remainder)

        batch_input_ids = []
        batch_attn = []
        batch_mask = []  # choice_mask: 프롬프트 제외(0), 선택지 부분(1)
        batch_labels = []

        for f in features:
            prompt_lens = f["prompt_lens"]  # len C
            ids_c = []
            attn_c = []
            mask_c = []
            for j in range(C):
                seq = f["input_ids"][j]
                attn = f["attention_mask"][j]
                # pad
                pad_len = max_len - len(seq)
                ids_c.append(seq + [self.tokenizer.pad_token_id] * pad_len)
                attn_c.append(attn + [0] * pad_len)
                # mask: 0..prompt_len-1 → 0, prompt_len..end → 1
                m = [0]*min(prompt_lens[j], max_len) + [1]*max(0, len(seq)-prompt_lens[j])
                # 패딩 구간은 0
                if len(m) < max_len:
                    m = m + [0]*(max_len - len(m))
                mask_c.append(m)

            batch_input_ids.append(ids_c)
            batch_attn.append(attn_c)
            batch_mask.append(mask_c)
            batch_labels.append(f["labels"])

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),        # (B, C, L)
            "attention_mask": torch.tensor(batch_attn, dtype=torch.long),        # (B, C, L)
            "choice_mask": torch.tensor(batch_mask, dtype=torch.long),           # (B, C, L)
            "labels": torch.tensor(batch_labels, dtype=torch.long),              # (B,)
        }


class CausalMCWrapper(nn.Module):
    def __init__(self, base_causal_lm):
        super().__init__()
        self.base = base_causal_lm
        self.config = getattr(base_causal_lm, "config", None)

        # 베이스 모델 forward가 받는 파라미터 집합 캐싱 (성능/안전)
        self._base_sig = inspect.signature(self.base.forward)
        self._base_allowed = set(self._base_sig.parameters.keys())

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        choice_mask=None,
        labels=None,
        **kwargs,
    ):
        # --- 1) 베이스로 보낼 kwargs 정제 ---
        # wrapper 전용/콜레이터 보조 키는 제거
        drop_keys = {"num_items_in_batch", "labels", "choice_mask", "prompt_lens"}
        for k in drop_keys:
            kwargs.pop(k, None)

        # 베이스 forward 시그니처에 존재하는 키만 통과
        base_kwargs = {k: v for k, v in kwargs.items() if k in self._base_allowed}

        # --- 2) (B,C,L) -> (B*C,L) 펼치기 ---
        B, C, L = input_ids.shape
        flat_ids = input_ids.view(B * C, L)
        flat_attn = attention_mask.view(B * C, L)
        flat_m = choice_mask.view(B * C, L)

        # --- 3) 베이스 호출 (불필요 kwargs 제거된 상태) ---
        out = self.base(input_ids=flat_ids, attention_mask=flat_attn, **base_kwargs)

        # --- 4) 선택지 점수 계산 ---
        logits = out.logits[:, :-1, :]                  # (B*C, L-1, V)
        targets = flat_ids[:, 1:]                        # (B*C, L-1)
        m = (flat_m[:, 1:] * flat_attn[:, 1:]).bool()    # (B*C, L-1)

        logprobs = F.log_softmax(logits, dim=-1)
        tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B*C, L-1)

        tok_lp = tok_lp.masked_fill(~m, 0.0)
        choice_scores = tok_lp.view(B, C, -1).sum(dim=-1).to(logits.dtype) # (B, C)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(choice_scores, labels)

        return {"loss": loss, "logits": choice_scores}


def get_target_module(model):
    """
    e.g.,
    target_modules = ["q", "k", "v", "o",     # attention (self-attention, encoder-decoder cross-attention)
                      "wi_0", "wi_1", "wo"    # mlp
                      ]
    """
    model_modules = str(model.modules)
    pattern = r'\((\w+)\): Linear'
    linear_layer_names = re.findall(pattern, model_modules)

    names = []
    # Print the names of the Linear layers
    for name in linear_layer_names:
        if "head" in name:
            continue
        names.append(name)
    target_modules = list(set(names))

    return target_modules


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_config(args):
    if "EXAONE" in args.model_name:
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(args.model_name)
    config.use_cache = False
    return config


def get_model(args):
    config = get_config(args)
    tokenizer = get_tokenizer(args)
    config.pad_token_id = tokenizer.pad_token_id

    if args.task_name in ["KB_COPA", "KB_HellaSwag"]:
        base = AutoModelForCausalLM.from_pretrained(
            args.model_name, config=config, trust_remote_code=True
        )
        base.resize_token_embeddings(len(tokenizer))
        return config, base, tokenizer
    else:
        if args.task_name == "KorSTS":
            config.num_labels = 1
            config.problem_type = "regression"
            model_cls = AutoModelForSequenceClassification
        elif args.task_name in ["NSMC", "PAWS_X", "KB_BoolQ", "KB_WiC", "KB_SentiNeg"]:
            config.num_labels = 2
            config.problem_type = "single_label_classification"
            model_cls = AutoModelForSequenceClassification
        elif args.task_name == "KorNLI":
            config.num_labels = 3
            config.problem_type = "single_label_classification"
            model_cls = AutoModelForSequenceClassification
        else:
            raise ValueError("Unknown task_name")

        model = model_cls.from_pretrained(args.model_name, config=config, trust_remote_code=True, use_safetensors=True)
        model.resize_token_embeddings(len(tokenizer))
        return config, model, tokenizer



def get_trainer(args, model, tokenizer, data_collator, train_dataset, val_dataset, compute_metrics):
    is_causal_mc = True if args.task_name in ["KB_COPA", "KB_HellaSwag"] else False

    if args.lora:
        task_type = TaskType.CAUSAL_LM if is_causal_mc else TaskType.SEQ_CLS

        print(f"Applying LoRA to model...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            task_type=task_type,
            target_modules="all-linear",
            bias="none",
        )

        print("\n* Applying LoRA to model...")
        print_trainable_parameters(model, prefix="BEFORE")
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model, prefix="AFTER")

        if args.gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            if hasattr(model, "config"):
                model.config.use_cache = False

    
    if args.set_script:
        trans_config = model.config
        trans_config.update({"is_cross_attention": False})
        trans_config.update({"embedding_norm": args.script_embedding_norm})

        if args.script_hidden_dim != trans_config.hidden_size:
            print(f"\n\n[BEFORE] script hidden dim: {args.script_hidden_dim}")
            print(f"[BEFORE] script intermediate dim: {args.script_intermediate_size}")
            args.script_hidden_dim = trans_config.hidden_size
            print(f"[AFTER] script hidden dim: {args.script_hidden_dim}")
            print(f"[AFTER] script intermediate dim: {args.script_intermediate_size}\n\n")
            # args.script_intermediate_size = trans_config.hidden_size

        # if args.script_num_attention_heads != trans_config.num_key_value_heads:
        #     args.script_num_attention_heads = trans_config.num_key_value_heads

        script_config = script_Config(
            tok_type=args.script_tok_type,
            reducer=args.script_reducer,
            hidden_dim=args.script_hidden_dim,
            script_max_length=args.script_max_length,
            max_length=args.max_input_length,
            do_combination=args.script_do_combination,
            combination_type=args.script_combination_type,
            trans_config=trans_config,
            num_attention_heads=args.script_num_attention_heads,
            intermediate_size=args.script_intermediate_size,
            num_trans_layers=args.script_num_trans_layers,
            dropout=args.script_dropout,
            is_bert=False,
            fusion=args.script_fusion,
        )
        
        if args.script_tok_type == "same":       # this option indicates the same tokenizer with PLM's
            script_tokenizer = tokenizer
            args.script_max_length = args.max_input_length
        else:
            script_tokenizer = get_gpt2_tokenizer(
                tok_type=args.script_tok_type,
                lang=args.script_lang,
                max_length=args.script_max_length,
                lowercase=True,
                clean_text=True,
                add_bos_token=False,
                bos_token='<|endoftext|>',
                eos_token='<|endoftext|>',
                unk_token='<unk>',
                pad_token='<|endoftext|>',
            )
            if (hasattr(script_tokenizer, "trunc_num") and
                    args.script_tok_type in ["jamo_var", "stroke_var", "cji_var", "bts_var"] and
                    args.script_max_length % script_tokenizer.trunc_num != 0):
                args.script_max_length = args.script_max_length - (args.script_max_length % script_tokenizer.trunc_num)
                script_tokenizer.max_length = args.script_max_length

                print(f"Change the max_length to {args.script_max_length} for the script_tokenizer's truncation.")

            # if args.task_name in ["KB_COPA", "KB_HellaSwag"] and (script_tokenizer.cls_token is None):
            #     _ = script_tokenizer.add_special_tokens({"cls_token": "[CLS]"})

        print_trainable_parameters(model, prefix="BEFORE script")
        model = apply_script_to_model(model, tokenizer, script_tokenizer, script_config)
        make_only_script_and_lora_as_trainable(model, weight='script_lora_only', bias='script_lora_only')
        print_trainable_parameters(model, prefix="AFTER script")
        print("\n\n")

    if is_causal_mc:
        wrapped = CausalMCWrapper(model)
        wrapped.config = getattr(model, "config", None)
        model = wrapped

    if args.task_name == "KorSTS":
        metric_for_best = "pearson"
    else:
        metric_for_best = "accuracy"

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=True if args.ckpt_dir else False,
        # if you want to continue the training from the checkpoint, you activate this and set the ckpt path using 'resume_from_checkpoint'
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed=args.deepspeed_config_file,
        torch_compile=args.compile,
        dataloader_num_workers=args.num_workers,

        remove_unused_columns=False,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        optim=args.optimizer,
        learning_rate=args.base_lr,
        lr_scheduler_type=args.lr_scheduler,  # constant, linear, cosine

        gradient_checkpointing=False,       # if it is True, the Qwen model does not work, raising the Error "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"

        report_to='tensorboard',
        logging_dir=args.tb_dir,
        logging_strategy=args.log_strategy,     # steps
        logging_steps=args.log_steps,
        logging_first_step=False,

        eval_strategy=args.eval_strategy,
        # eval_steps=args.eval_steps,
        eval_on_start=args.debugging,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_accumulation_steps=args.eval_grad_acc,

        load_best_model_at_end=False,
        metric_for_best_model=metric_for_best,
        greater_is_better=True,

        save_strategy=args.save_strategy,       # steps or epochs
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    )
    trainer = Trainer(
        args=training_args,
        model=model,
        # tokenizer=tokenizer,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="dummy")
    parser.add_argument("--host", type=str, default="dummy")
    parser.add_argument("--port", type=int, default=5678)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--num_processes", type=int)
    parser.add_argument("--num_gpus", type=int)

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--fp16", type=_bool, default=False, help="Enable 16-bit floating-point mixed precision")
    parser.add_argument("--bf16", type=_bool, default=True, help="Enable bfloat16 mixed precision")
    parser.add_argument("--compile", type=_bool, default=True)
    parser.add_argument("--debugging", type=_bool, default=False, help="Enable debugging mode. If it's true, 'eval_on_start' would be activated.")
    parser.add_argument("--is_toyset", type=_bool, default=False, help="Indicates whether the dataset is a toyset or not")

    parser.add_argument("--deepspeed", type=_bool, default=True, help="Enable DeepSpeed optimization")
    parser.add_argument("--deepspeed_stage", type=int, default=2, help="DeepSpeed stage for ZeRO optimization")
    parser.add_argument("--deepspeed_offload", type=_bool, default=False, help="Enable DeepSpeed optimizer offloading")
    parser.add_argument("--deepspeed_config_file", type=str)

    parser.add_argument("--lora", type=_bool, default=True)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.03)

    # script configuration
    parser.add_argument('--set_script', type=_bool, default=False)
    parser.add_argument('--script_tok_type', type=str, default="jamo_var")
    parser.add_argument('--script_lang', type=str, default="ko")
    parser.add_argument('--script_reducer', type=str, default="linear")
    parser.add_argument('--script_hidden_dim', type=int, default=768)
    parser.add_argument('--script_max_length', type=int, default=2048)
    parser.add_argument('--script_do_combination', type=_bool, default=True)
    parser.add_argument('--script_embedding_norm', type=_bool, default=False)
    parser.add_argument('--script_combination_type', type=str, default="gru")
    parser.add_argument('--script_intermediate_size', type=int, default=768)
    parser.add_argument('--script_num_attention_heads', type=int, default=2)
    parser.add_argument('--script_dropout', type=float, default=0.02)
    parser.add_argument('--script_num_trans_layers', type=int, default=3)         # it activates when "trans" is in the "script_combination_type" var.
    parser.add_argument('--script_fusion', type=str, default="cross_attn")
    parser.add_argument('--script_add_lora', type=_bool, default=False)
    parser.add_argument('--script_lora_r', type=int, default=32)
    parser.add_argument('--script_lora_alpha', type=int, default=128)
    parser.add_argument('--script_lora_dropout', type=float, default=0.03)

    # Dataset Configuration
    parser.add_argument("--task_name", type=str, default="KorSTS",
                        help="KorNLI  ||  KorSTS  ||  NSMC  ||  PAWS_X  ||  "
                             "KB_BoolQ  ||  KB_COPA  ||  KB_WiC  ||  KB_HellaSwag  ||  KB_SentiNeg")

    # NLU dataset configuration
    parser.add_argument("--remain_lang", type=str, default="ko_punc")
    parser.add_argument("--do_hangeulize", type=_bool, default=False)
    parser.add_argument("--data_remove", type=_bool, default=False)

    parser.add_argument("--model_name", type=str, default="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
                        help="EleutherAI/polyglot-ko-1.3b  ||  skt/ko-gpt-trinity-1.2B-v0.5  ||  "
                             "Qwen/Qwen2.5-3B  ||  Qwen/Qwen2.5-7B  ||  "
                             "yanolja/EEVE-Korean-10.8B-v1.0  ||  yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
                             "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct  ||  LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
                             "LGAI-EXAONE/EXAONE-4.0-1.2B")

    parser.add_argument("--ckpt_dir", type=str, help="If you want to continue training from checkpoint in trainer, set this option.")
    parser.add_argument("--peft_model_location", type=str,
                        default="",
                        help="If you want to load the specific fine-tuned PEFT model, set this option. You should enter the checkpoint directory e.g., checkpoint-130000")
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--optimizer", type=str, default="adamw_torch")
    parser.add_argument("--base_lr", type=float, default=5e-05)         # 5e-05
    parser.add_argument("--batch_size", type=int, default=8)            # 16
    parser.add_argument("--grad_acc", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)               # 4
    parser.add_argument("--lr_scheduler", type=str, default="linear", help="linear  ||  cosine  || constant")
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--final_cosine", type=float, default=1e-5)
    parser.add_argument("--gradient_checkpointing", type=_bool, default=True)

    # Model save-related arguments
    parser.add_argument("--save_strategy", type=str, default="epoch", help="epoch  ||  steps")
    parser.add_argument("--save_steps", type=int, default=315)           # 10_000
    parser.add_argument("--save_total_limit", type=int, default=100)

    # Evaluation-related arguments
    parser.add_argument("--eval_strategy", type=str, default="epoch", help="epoch  ||  steps")
    parser.add_argument("--eval_steps", type=int, default=5_000)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_grad_acc", type=int, default=1)
    # logging arguments
    parser.add_argument("--log_strategy", type=str, default="steps")
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--tb_dir", type=str)
    parser.add_argument("--save_dir", type=str)

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    specific_model_type = ""
    if args.lora:
        specific_model_type += "lora_"
    if args.set_script:
        specific_model_type += f"script_{args.script_tok_type}_{args.script_max_length}t_"
        if args.script_do_combination:
            specific_model_type += f"comb-{args.script_combination_type}_"
            if args.script_fusion == "cross_attn":
                specific_model_type += f"{args.script_intermediate_size}idim_"
        else:
            specific_model_type += f"red-{args.script_reducer}_"
        specific_model_type += f"{args.script_fusion}_"

    args.log_dir = os.path.join(f"logs/{args.model_name.replace('/', '_')}/nlu_tasks/{args.task_name}/{specific_model_type}{args.max_input_length}t_{args.batch_size}b_{args.grad_acc}s_{args.epochs}ep_{args.base_lr}lr_{args.seed}rs")

    if args.compile:
        args.log_dir += "_compile"
    if args.deepspeed:
        args.log_dir += f"_ds-stage{args.deepspeed_stage}"
        if args.deepspeed_offload:
            args.log_dir += "-offload"
    if args.fp16:
        args.log_dir += "_fp16"
    if args.bf16:
        args.log_dir += "_bf16"


    args.save_dir = os.path.join(args.log_dir, "ckpt")
    args.tb_dir = os.path.join(args.log_dir, "tb")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.tb_dir, exist_ok=True)

    if args.deepspeed:
        args.deepspeed_config_file = f"configs/ds_zero{args.deepspeed_stage}"
        if args.deepspeed_offload:
            args.deepspeed_config_file += "_offload"
        args.deepspeed_config_file += "_config.json"
    else:
        args.deepspeed_config_file = None

    args_dict = args.__dict__
    json.dump(args_dict, open(f"{args.log_dir}/args.json", 'w'), indent=2)

    init_random(args.seed)


    print("\n* Loading the Model...")
    config, model, tokenizer = get_model(args)
    print("Done.\n")




    print("* Loading the Dataset...")
    total_dataset = get_nlu_dataset(args, tokenizer)
    train_dataset = total_dataset["train"]
    # val_dataset = total_dataset["dev"]
    val_dataset = total_dataset["test"]

    print("Done.\n")

    print("* Loading the DataCollator...")
    if args.task_name in ["KB_COPA", "KB_HellaSwag"]:
        data_collator = DataCollatorForCausalMC(tokenizer=tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
        if args.task_name == "KorSTS":
            @dataclass
            class CollatorFloatLabels:
                base: DataCollatorWithPadding

                def __call__(self, features):
                    batch = self.base(features)
                    if "labels" in batch:
                        batch["labels"] = batch["labels"].to(torch.float32)  # 또는 model.dtype
                    return batch

            data_collator = CollatorFloatLabels(data_collator)

    print("Done.\n")

    if args.task_name == "KorSTS":
        def compute_metrics(p):
            preds = p.predictions.reshape(-1)  # (N,1) → (N,)
            labels = p.label_ids.reshape(-1).astype(np.float32)
            pear = pearsonr(preds, labels).statistic
            spear = spearmanr(preds, labels).statistic
            mse = np.mean((preds - labels) ** 2)
            return {"pearson": float(pear), "spearman": float(spear), "mse": float(mse)}
    else:
        acc = evaluate.load("accuracy")

        def compute_metrics(p):
            preds = p.predictions.argmax(-1)
            labels = p.label_ids
            return {"accuracy": acc.compute(predictions=preds, references=labels)["accuracy"]}



    print("* Getting the Trainer...")
    trainer = get_trainer(args, model, tokenizer, data_collator, train_dataset, val_dataset, compute_metrics=compute_metrics)
    print("Done.\n")

    if args.ckpt_dir:
        trainer.train(resume_from_checkpoint=args.ckpt_dir)
    else:
        trainer.train()

    print()
    print("Finished")
    # trainer.save_model(os.path.join(args.save_dir, "checkpoint-final"))
    # print("Saved model")


if __name__ == "__main__":
    main()
