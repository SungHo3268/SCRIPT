import os
import sys
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any
from types import SimpleNamespace as NS

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cpu\.amp\.autocast.*"
)

import torch
import numpy as np
from datasets import Dataset

from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, DataCollatorForSeq2Seq,
    GenerationConfig, TrainingArguments, Trainer,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    PreTrainedTokenizerFast
)
import transformers
transformers.logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
warnings.filterwarnings("ignore", message=".*Some weights of.*were not used.*")

from torch.utils.data import DataLoader
from accelerate import Accelerator

from peft import LoraConfig, get_peft_model, TaskType

# --- project-local imports ---
sys.path.append(os.getcwd())
from srcs.functions import init_random
from srcs.gpt_utils import text_tokenization_for_casuallm
from pretraining.scripts.run_gpt_pretraining import get_gpt2_tokenizer
from srcs.script import make_only_script_and_lora_as_trainable, apply_script_to_model, script_Config

# legacy metrics
from nlg_tasks.srcs.evaluation_metrics import eval_main
from utils.m2scorer.scripts.m2scorer import m2score_main
from utils.gleu_scorer.gleumodule import run_gleu

# 참고코드 2의 커스텀 트레이너
from nlg_tasks.srcs.trainer import GPTNLGTrainer


# ==========================
# Small utils
# ==========================

def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, train


def print_trainable_parameters(model, prefix=None):
    tot0, tr0 = count_params(model)
    msg = f"total={tot0:,}, trainable={tr0:,}"
    if prefix: msg = f"[{prefix}] {msg}"
    print(msg)


def set_logger(args, tqdm_handler=True):
    from utils.logger import get_logger

    logger = get_logger(log_path=os.path.join(args.log_dir, "train_log.txt"), tqdm_handler=tqdm_handler)
    logger.info("\n")
    logger.info("This train_log.txt inform the Running Progress.\n")
    logger.info(f"Save the parser information to {args.log_dir}")

    logger.info("\n")
    logger.info(f"Arguments: {args}\n")

    # with open(os.path.join(args.logging.log_dir, 'argparse.json'), 'w') as fw:
    #     json.dump(dict(args), fw, indent=2)
    #     fw.close()
    return logger


# ==========================
# Dataset builders
# ==========================

def get_nlg_dataset(args, tokenizer):
    """
    HF datasets (map → DatasetDict) for HF Trainer 경로
    """
    if args.task_name == 'KoCommonGen':
        task_util_path = "nlg_tasks.data_utils.KoCommonGen.data_utils"
    elif args.task_name == 'XL_Sum':
        task_util_path = "nlg_tasks.data_utils.XL_Sum.data_utils"
    elif 'KoreanGEC' in args.task_name:
        task_util_path = "nlg_tasks.data_utils.KoreanGEC.data_utils"
    else:
        raise ValueError("Unknown NLG task_name. Choose among [KoCommonGen, XL_Sum, KoreanGEC*]")

    import importlib
    task_util = importlib.import_module(task_util_path, package=".")
    dataset = task_util.load_task_dataset()
    if 'KoreanGEC' in args.task_name:
        dataset = dataset[args.task_name]

    total_dataset = {}
    for mode in ['train', 'dev', 'test']:
        data = Dataset.from_dict(dataset[mode])
        bsz = args.batch_size // args.grad_acc if mode == 'train' else args.eval_batch_size
        tokenized = data.map(
            text_tokenization_for_casuallm,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": args.max_input_length + args.max_target_length + 5,
                "max_new_tokens": args.max_target_length,
                "task_name": args.task_name,
                "mode": mode,
            },
            remove_columns=data.column_names,
            batched=True,
            batch_size=bsz,
        )
        total_dataset[mode] = tokenized
    return total_dataset


def get_nlg_dataloaders_for_accelerate(args, tokenizer):
    """
    참고코드 1의 DataLoader 버전을 현재 argparse/구조에 맞춰 구성.
    """
    # dataset 로드
    if args.task_name == 'KoCommonGen':
        from nlg_tasks.data_utils.KoCommonGen.data_utils import load_task_dataset
    elif args.task_name == 'XL_Sum':
        from nlg_tasks.data_utils.XL_Sum.data_utils import load_task_dataset
    elif 'KoreanGEC' in args.task_name:
        from nlg_tasks.data_utils.KoreanGEC.data_utils import load_task_dataset
    else:
        raise ValueError("Unknown task_name")

    dataset = load_task_dataset()
    if 'KoreanGEC' in args.task_name:
        dataset = dataset[args.task_name]

    # 필요 시 샘플링/슬라이스를 원하면 여기서 조절 가능
    total_dataloader = dict()
    for mode in ['train', 'dev', 'test']:
        data = Dataset.from_dict(dataset[mode])
        bsz = args.batch_size // args.grad_acc if mode == 'train' else args.eval_batch_size
        tokenized = data.map(
            text_tokenization_for_casuallm,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": args.max_input_length,
                "max_new_tokens": args.max_target_length,
                "task_name": args.task_name,
                "mode": mode,
            },
            remove_columns=data.column_names,
            batched=True,
            batch_size=bsz,
        )
        if mode == 'train':
            collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
            shuffle = True
        else:
            collator = DataCollatorForSeq2Seq(tokenizer)
            shuffle = False
        dl = DataLoader(
            tokenized,
            shuffle=shuffle,
            collate_fn=collator,
            batch_size=bsz,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        total_dataloader[mode] = dl
    return total_dataloader


# ==========================
# Model / Tokenizer
# ==========================
def get_tokenizer(args):
    if 'skt/kogpt2' in args.model_name:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            args.model_name,
            bos_token='</s>', eos_token='</s>', unk_token='<unk>',
            pad_token='<pad>', mask_token='<mask>',
            padding_side='left'
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side='left')
    return tokenizer


def get_causallm(args):
    tokenizer = get_tokenizer(args)
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.use_cache = False
    if "skt/kogpt2" in args.model_name:
        config.tie_word_embeddings = False
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, config=config, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    return config, model, tokenizer


def maybe_wrap_lora(args, model):
    if not args.lora:
        return model
    print("Applying LoRA to model…")
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
        bias="none",
    )
    print_trainable_parameters(model, prefix="BEFORE LoRA")
    model = get_peft_model(model, lora_cfg)
    print_trainable_parameters(model, prefix="AFTER  LoRA")
    return model


def maybe_apply_script(args, model, plm_tokenizer):
    if not args.set_script:
        return model

    trans_config = model.config
    trans_config.update({"is_cross_attention": False})
    trans_config.update({"embedding_norm": args.script_embedding_norm})

    if args.script_hidden_dim != trans_config.hidden_size:
        print(f"[script] override hidden_dim {args.script_hidden_dim} -> {trans_config.hidden_size}")
        args.script_hidden_dim = trans_config.hidden_size

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

    if args.script_tok_type == "same":
        script_tok = plm_tokenizer
        args.script_max_length = args.max_input_length
    else:
        script_tok = get_gpt2_tokenizer(
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
        if (hasattr(script_tok, "trunc_num") and
                args.script_tok_type in ["jamo_var", "stroke_var", "cji_var", "bts_var"] and
                args.script_max_length % script_tok.trunc_num != 0):
            args.script_max_length = args.script_max_length - (args.script_max_length % script_tok.trunc_num)
            script_tok.max_length = args.script_max_length
            print(f"[script] Adjust script_max_length -> {args.script_max_length} (align truncation)")

    print_trainable_parameters(model, prefix="BEFORE script")
    model = apply_script_to_model(model, plm_tokenizer, script_tok, script_config)
    make_only_script_and_lora_as_trainable(model, weight='script_lora_only', bias='script_lora_only')
    print_trainable_parameters(model, prefix="AFTER  script")
    return model


# ==========================
# HF Trainer subclass to capture inputs during eval
# ==========================
class TextGenTrainer(Seq2SeqTrainer):
    def evaluation_loop(self, dataloader, *args, **kwargs):
        self._stored_inputs = {"input_ids": [], "attention_mask": []}
        self._metric_mode = kwargs.get("metric_key_prefix", "eval")
        return super().evaluation_loop(dataloader, *args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        if not prediction_loss_only and ("input_ids" in inputs):
            self._stored_inputs["input_ids"].append(inputs["input_ids"].detach().cpu())
            if "attention_mask" in inputs:
                self._stored_inputs["attention_mask"].append(inputs["attention_mask"].detach().cpu())
        return loss, logits, labels


# ==========================
# Legacy metrics wrapper (eval_main, m2scorer, GLEU)
# ==========================
class GenerationEvaluator:
    def __init__(self, tokenizer, task_name, datasets_root="datasets/nlg_tasks"):
        self.tok = tokenizer
        self.task_name = task_name
        self.datasets_root = datasets_root

    @staticmethod
    def _to_2d_int_lists(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
        except Exception:
            pass

        if isinstance(x, tuple) and len(x) == 1:
            x = x[0]

        if isinstance(x, np.ndarray):
            if x.dtype == object or x.ndim == 1:
                seqs = []
                for row in x:
                    row = row.tolist() if isinstance(row, np.ndarray) else row
                    seqs.append(list(row))
            elif x.ndim >= 2:
                seqs = x.tolist()
            else:
                seqs = [x.tolist()]
        elif isinstance(x, (list, tuple)):
            seqs = [r.tolist() if hasattr(r, "tolist") else list(r) for r in x]
        else:
            seqs = [list(x)]

        if seqs and isinstance(seqs[0], (list, np.ndarray)) and seqs[0] and isinstance(seqs[0][0], (list, np.ndarray)):
            seqs = [s[0] for s in seqs]

        out = []
        for s in seqs:
            out.append([int(t) if isinstance(t, (int, np.integer)) else 0 for t in s])
        return out

    def _decode(self, arr):
        seqs = self._to_2d_int_lists(arr)
        pad = self.tok.pad_token_id
        if pad is None:
            pad = self.tok.eos_token_id

        vmax = len(self.tok) - 1
        vmin = 0
        safe = []
        for s in seqs:
            ss = []
            for t in s:
                if t == -100 or t < vmin or t > vmax:
                    ss.append(pad)
                else:
                    ss.append(t)
            safe.append(ss)

        return [s.strip() for s in self.tok.batch_decode(
            safe, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )]

    def _extract_only_predictions(self, full_preds, given_texts):
        only = []
        for pred, g in zip(full_preds, given_texts):
            tail = pred[len(g):].strip() if pred.startswith(g) else pred
            if self.task_name in ['KoCommonGen', 'XL_Sum'] or 'KoreanGEC' in self.task_name:
                if self.task_name == 'KoCommonGen':
                    end = tail.find('.')
                elif self.task_name == 'XL_Sum':
                    cand = [x for x in [tail.find('.'), tail.find('?')] if x != -1]
                    end = min(cand) if cand else tail.find('.')
                elif 'KoreanGEC' in self.task_name:
                    cand = [tail.find(x) for x in ['.', '?', '!', '\n', '\t']]
                    cand = [c for c in cand if c != -1]
                    end = min(cand) if cand else -1
                else:
                    end = -1
                if end == 0:
                    tail = tail[1:]
                    end = tail.find('.')
                    if (end == -1) or (end > len(tail) - 1):
                        end = len(tail) - 1
                elif end == -1:
                    end = len(tail) - 1
                tail = tail[: end + 1].strip()
            else:
                raise NotImplementedError
            only.append(tail)
        return only

    def __call__(self, eval_preds, stored_inputs, mode="dev"):
        pred_ids = eval_preds.predictions[0] if isinstance(eval_preds.predictions,
                                                           (list, tuple)) else eval_preds.predictions
        label_ids = eval_preds.label_ids

        pred_full = self._decode(pred_ids)
        refs = self._decode(label_ids)

        given_texts = []
        for batch_ids in stored_inputs["input_ids"]:
            given_texts.extend(self._decode(batch_ids))

        only_preds = self._extract_only_predictions(pred_full, given_texts)

        concepts_list = None
        refs_for_metric = refs
        if self.task_name == 'KoCommonGen':
            concepts = [[m.strip() for m in s.split(",")] for s in given_texts]
            concepts_list = ["#".join(c) for c in concepts]
            if mode == 'test':
                refs_for_metric = [r.split(' = ') for r in refs]
        elif 'KoreanGEC' in self.task_name:
            refs_plain = refs
            if self.task_name == 'KoreanGEC_union':
                clipped = []
                for pred, ref in zip(only_preds, refs_plain):
                    clipped.append(pred[:len(ref) * 3 // 2] if len(pred) > len(ref) * 1.5 else pred)
                only_preds = clipped
            parts = self.task_name.split('_')
            path_1 = parts[0]
            path_2 = '_'.join(parts[1:]) if len(parts) > 1 else parts[0]
            eval_mode = 'val' if mode == 'dev' else 'test'
            m2_path = f"{self.datasets_root}/{path_1}/{path_2}/{path_2}_{eval_mode}.m2"
            p, r, f1 = m2score_main(only_preds, m2_path)
            gleu = float(run_gleu(refs_plain, given_texts, only_preds))
            return {'m2_precision': p, 'm2_recall': r, 'm2_f1_half': f1, 'gleu': gleu}

        if isinstance(refs_for_metric[0], str):
            refs_for_metric = [[r] for r in refs_for_metric]
        results = eval_main(refs_for_metric, only_preds, concepts_list)['total_avg']
        return results


# ==========================
# Main
# ==========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="dummy")
    parser.add_argument("--host", type=str, default="dummy")
    parser.add_argument("--port", type=int, default=5678)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--num_processes", type=int)
    parser.add_argument("--num_gpus", type=int)

    # Backend 선택
    parser.add_argument("--trainer_backend", type=str, default="hf",
                        choices=["hf", "accelerate"],
                        help="[hf] HuggingFace Trainer | [accelerate] GPTNLGTrainer")

    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--bf16", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--compile", type=lambda x: x.lower() == 'true', default=False)

    # Dist / DS (HF 경로에서만 사용)
    parser.add_argument("--deepspeed", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--deepspeed_stage", type=int, default=2)
    parser.add_argument("--deepspeed_offload", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--deepspeed_config_file", type=str)

    # LoRA
    parser.add_argument("--lora", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # script
    parser.add_argument('--set_script', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--script_tok_type', type=str, default="jamo_var")
    parser.add_argument('--script_lang', type=str, default="ko")
    parser.add_argument('--script_reducer', type=str, default="linear")
    parser.add_argument('--script_hidden_dim', type=int, default=768)
    parser.add_argument('--script_max_length', type=int, default=2048)
    parser.add_argument('--script_do_combination', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--script_embedding_norm', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--script_combination_type', type=str, default="gru")
    parser.add_argument('--script_intermediate_size', type=int, default=768)
    parser.add_argument('--script_num_attention_heads', type=int, default=2)
    parser.add_argument('--script_dropout', type=float, default=0.1)
    parser.add_argument('--script_num_trans_layers', type=int, default=3)
    parser.add_argument('--script_fusion', type=str, default="cross_attn")

    # Task / Data
    parser.add_argument("--task_name", type=str, default="KoCommonGen",
                        help="KoCommonGen || XL_Sum || KoreanGEC*")
    parser.add_argument("--model_name", type=str, default="skt/kogpt2-base-v2",
                        help="skt/kogpt2-base-v2  || skt/ko-gpt-trinity-1.2B-v0.5 || LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct")
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)

    # Generation Config
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)

    # Train/Eval
    parser.add_argument("--optimizer", type=str, default="adamw_torch",
                        help="[HF] adamw_torch 등 | [accelerate] adamw/adamwscale/adafactor")
    parser.add_argument("--base_lr", type=float, default=2e-2)
    parser.add_argument("--weight_decay", type=float, default=0.01)  # accelerate 경로에서 사용
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_acc", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr_scheduler", type=str, default="linear",
                        help="[HF] linear/cosine 등 | [accelerate] linear/cosine/legacy/constant")
    parser.add_argument("--final_cosine", type=float, default=1e-5)  # accelerate cosine용
    parser.add_argument("--warmup_ratio", type=float, default=0.06)  # accelerate 경로에서 total_steps 기반
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_grad_acc", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=0)

    # Logging / Saving`
    parser.add_argument("--log_strategy", type=str, default="steps")
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=50)
    parser.add_argument("--log_grad_l2", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--log_weights_l2", type=lambda x: x.lower() == 'true', default=False)
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Directories
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

    args.log_dir = os.path.join(
        f"logs/{args.model_name.replace('/', '_')}/nlg_tasks/{args.task_name}/{specific_model_type}{args.max_input_length}+{args.max_target_length}t_{args.batch_size}b_{args.grad_acc}s_{args.epochs}ep_{args.base_lr}lr_{args.seed}rs"
    )
    if args.fp16:
        args.log_dir += "_fp16"
    if args.bf16:
        args.log_dir += "_bf16"

    args.save_dir = os.path.join(args.log_dir, "ckpt")
    args.tb_dir = os.path.join(args.log_dir, "tb")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.tb_dir, exist_ok=True)

    # build deepspeed config path like NLU (HF 경로에서만)
    if args.trainer_backend == "hf" and args.deepspeed:
        args.deepspeed_config_file = f"configs/ds_zero{args.deepspeed_stage}"
        if args.deepspeed_offload:
            args.deepspeed_config_file += "_offload"
        args.deepspeed_config_file += "_config.json"
    else:
        args.deepspeed_config_file = None

    json.dump(args.__dict__, open(f"{args.log_dir}/args.json", 'w'), indent=2)

    init_random(args.seed)

    # Load model / tokenizer
    print("\n* Loading the Causal LM model…")
    config, model, tokenizer = get_causallm(args)
    print("Done.\n")


    # Apply adapters
    model = maybe_wrap_lora(args, model)
    model = maybe_apply_script(args, model, tokenizer)

    # Generation config
    gen_conf = GenerationConfig.from_model_config(model.config)
    gen_conf.max_new_tokens = args.max_target_length
    gen_conf.num_beams = args.num_beams
    gen_conf.do_sample = False
    gen_conf.eos_token_id = tokenizer.eos_token_id
    gen_conf.pad_token_id = tokenizer.pad_token_id

    model.generation_config = gen_conf


    # ------------------------------------------------------------------
    # Backend 분기: HF Trainer  vs  Accelerate + GPTNLGTrainer
    # ------------------------------------------------------------------
    if args.trainer_backend == "hf":
        # Datasets & collators
        datasets = get_nlg_dataset(args, tokenizer)
        train_ds = datasets['train']
        # train_ds = train_ds.shuffle(seed=args.seed).select(range(min(5000, len(train_ds))))
        val_ds = datasets['dev']
        # val_ds = val_ds.shuffle(seed=args.seed).select(range(min(500, len(val_ds))))


        @dataclass
        class CausalLMTrainEvalCollator:
            tokenizer: Any
            label_pad_token_id: int = -100

            def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
                # 1) 먼저 input_ids/attention_mask를 pad
                batch = self.tokenizer.pad(
                    {k: [f[k] for f in features] for k in features[0].keys() if k != "labels"},
                    padding=True,
                    return_tensors="pt",
                )
                B, L = batch["input_ids"].shape

                if "labels" in features[0]:
                    # ---- 핵심: labels를 각 샘플의 input_ids 길이에 맞게 "재구성" ----
                    padded_labels = []
                    pad_side = getattr(self.tokenizer, "padding_side", "right")

                    for f in features:
                        ids_len = len(f["input_ids"])
                        lab = list(f["labels"])

                        # (A) labels 길이를 ids_len로 맞추되, 짧으면 suffix에 붙이고 앞은 -100
                        if len(lab) == ids_len:
                            full = lab
                        elif len(lab) < ids_len:
                            full = [self.label_pad_token_id] * (ids_len - len(lab)) + lab
                        else:  # len(lab) > ids_len
                            full = lab[-ids_len:]  # 마지막 ids_len만 사용

                        # (B) 배치 pad로 ids_len -> L 이 됐으니, pad_side에 맞춰 -100 추가
                        pad_len = L - ids_len
                        if pad_len < 0:
                            # 이 경우는 거의 없지만 안전하게 truncate
                            full = full[-L:]
                            pad_len = 0

                        if pad_side == "left":
                            full = [self.label_pad_token_id] * pad_len + full
                        else:
                            full = full + [self.label_pad_token_id] * pad_len

                        padded_labels.append(full)

                    batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

                else:
                    # train: labels 없으면 input_ids로 생성 (응급/기본)
                    labels = batch["input_ids"].clone()
                    pad_id = self.tokenizer.pad_token_id
                    if pad_id is not None:
                        labels[labels == pad_id] = self.label_pad_token_id
                    batch["labels"] = labels

                return batch


        collator = CausalLMTrainEvalCollator(tokenizer)

        # HF TrainingArguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.save_dir,
            # overwrite_output_dir=bool(args.ckpt_dir),
            seed=args.seed,
            fp16=args.fp16, bf16=args.bf16,
            deepspeed=args.deepspeed_config_file,
            torch_compile=args.compile,
            dataloader_num_workers=args.num_workers,
            remove_unused_columns=False,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_acc,
            optim=args.optimizer,
            learning_rate=args.base_lr,
            lr_scheduler_type=args.lr_scheduler,
            report_to='tensorboard',
            logging_dir=args.tb_dir,
            logging_strategy=args.log_strategy,
            logging_steps=args.log_steps,
            eval_strategy=args.eval_strategy,
            eval_steps=args.eval_steps,
            per_device_eval_batch_size=args.eval_batch_size,
            eval_accumulation_steps=args.eval_grad_acc,
            load_best_model_at_end=False,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            predict_with_generate=True,
            generation_num_beams=gen_conf.num_beams,
        )

        evaluator = GenerationEvaluator(tokenizer, args.task_name)

        trainer = TextGenTrainer(
            args=training_args,
            model=model,
            processing_class=tokenizer,
            data_collator=collator,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=lambda p: evaluator(p, trainer._stored_inputs, mode="dev"),
        )
        trainer._eval_data_collator = collator

        print("\nStart Training (HF Trainer + legacy metrics)…")
        if args.ckpt_dir:
            trainer.train(resume_from_checkpoint=args.ckpt_dir)
        else:
            trainer.train()
        print("Finished")

    else:
        # ---------------- Accelerate + GPTNLGTrainer 경로 ----------------
        # dataloaders
        dls = get_nlg_dataloaders_for_accelerate(args, tokenizer)

        # total_steps / warmup_steps 계산
        num_train_batches = len(dls['train'])
        total_steps = int((num_train_batches / max(1, args.grad_acc)) * args.epochs)
        warmup_steps = round(total_steps * args.warmup_ratio)

        # hparams 어댑터 (참고코드 2에서 쓰는 구조 맞춤)
        hparams = NS(
            device='gpu' if args.device == 'cuda' else 'cpu',
            seed=args.seed,
            logging=NS(
                log_dir=args.log_dir,
                save_dir=os.path.join(args.log_dir, "ckpt"),
                tb_dir=args.tb_dir,
                log_steps=args.log_steps,
                grad_l2=args.log_grad_l2,
                weights_l2=args.log_weights_l2,
            ),
            optim=NS(
                name=('adamw' if args.optimizer.startswith('adamw') else
                      ('adafactor' if 'adafactor' in args.optimizer else 'adamw')),
                base_lr=args.base_lr,
                weight_decay=args.weight_decay,
                lr_scheduler=args.lr_scheduler,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                final_cosine=args.final_cosine,
                epochs=args.epochs,
                grad_acc=args.grad_acc,
                train_batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                early_stop_patience=args.early_stop_patience,
                grad_clip=args.grad_clip,
            ),
            data=NS(
                task_name=args.task_name,
                num_workers=args.num_workers,
                max_length=args.max_input_length + args.max_target_length + 5,  # 참고코드 관례
            ),
            model=NS(
                name=args.model_name,
                hf_model=True,
                generation_config=dict(
                    max_length=args.max_input_length,
                    max_new_tokens=args.max_target_length,
                    do_sample=False,
                    num_beams=args.num_beams,
                    repetition_penalty=None,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    length_penalty=None,
                ),
                set_lora=bool(args.lora),
                set_script=bool(args.set_script),
                ckpt_dir=args.ckpt_dir,
                script=NS(
                    tok_type=args.script_tok_type,
                    lang=args.script_lang,
                    reducer=args.script_reducer,
                    hidden_dim=args.script_hidden_dim,
                    max_length=args.script_max_length,
                    embedding_norm=bool(args.script_embedding_norm),
                    do_combination=bool(args.script_do_combination),
                    combination=NS(
                        combination_type=args.script_combination_type,
                        num_attention_heads=args.script_num_attention_heads,
                        intermediate_size=args.script_intermediate_size,
                        num_trans_layers=args.script_num_trans_layers,
                    ),
                    fusion=args.script_fusion,
                ),
                lora=NS(
                    r=args.lora_r,
                    alpha=args.lora_alpha,
                    dropout=args.lora_dropout,
                ),
            ),
        )

        # generation_config를 실제 모델에 반영
        model.generation_config.max_length = hparams.model.generation_config["max_length"]
        model.generation_config.max_new_tokens = hparams.model.generation_config["max_new_tokens"]
        model.generation_config.do_sample = hparams.model.generation_config["do_sample"]
        model.generation_config.num_beams = hparams.model.generation_config["num_beams"]
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.bos_token_id = tokenizer.eos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

        # Accelerator 초기화 (bf16/fp16 반영)
        mixed_precision = 'bf16' if args.bf16 else ('fp16' if args.fp16 else 'no')
        accelerator = Accelerator(cpu=(args.device != 'cuda'), mixed_precision=mixed_precision)

        # 커스텀 트레이너 생성
        logger = set_logger(args, tqdm_handler=True)
        import logging
        logger.setLevel(logging.INFO)
        trainer = GPTNLGTrainer(hparams, accelerator, logger=logger, tokenizer=tokenizer, model=model, dataloaders=dls)

        # 로그 요약
        print("\n========== Fine-tuning with Accelerate + GPTNLGTrainer ==========")
        print(f"model              : {args.model_name}")
        print(f"task               : {args.task_name}")
        print(f"vocab size         : {len(tokenizer)}")
        print(f"device             : {args.device}")
        print(f"seed               : {args.seed}")
        print(f"epochs             : {args.epochs}")
        print(f"total steps        : {total_steps}")
        print(f"warmup steps       : {warmup_steps}")
        print(f"train batch size   : {args.batch_size} (accum {args.grad_acc})")
        print(f"eval batch size    : {args.eval_batch_size}")
        print(f"optimizer          : {hparams.optim.name}")
        print(f"lr scheduler       : {args.lr_scheduler}")
        print(f"learning rate      : {args.base_lr}")
        if args.lora:
            print(f"LoRA r/alpha/drop  : {args.lora_r}/{args.lora_alpha}/{args.lora_dropout}")
        if args.set_script:
            print(f"script tok_type/fusion: {args.script_tok_type}/{args.script_fusion}")
        print(f"num_beams          : {args.num_beams}")
        print(f"no_repeat_ngram_size : {args.no_repeat_ngram_size}")
        print(f"log dir            : {args.log_dir}")
        print(f"save dir           : {args.save_dir}")
        print(f"tb dir             : {args.tb_dir}\n")

        # 학습
        trainer.train()
        print("Finished (Accelerate path)")


if __name__ == "__main__":
    main()