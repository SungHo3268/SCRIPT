import os
import sys
from hydra import compose, initialize
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, GenerationConfig
from nlg_tasks.scripts.run_gpt_finetuning import get_config_and_nlg_model
sys.path.append(os.getcwd())
from utils.gen_utils import setup_basics
from nlg_tasks.srcs.trainer import GPTNLGTrainer
from pretraining.scripts.run_gpt_pretraining import get_gpt2_tokenizer
plt.rc('font', family='NanumBarunGothic')
mpl.rcParams['axes.unicode_minus'] = False

from matplotlib.colors import LinearSegmentedColormap
yellow_cmap = LinearSegmentedColormap.from_list("yellow_cmap", ["#FFFFE0", "#FFFF00", "#FFD700"])
yellow_cmap_adjusted = LinearSegmentedColormap.from_list("yellow_cmap_adjusted", ["#FFFFFF", "#FFFFE0", "#FFFF00", "#FFD700"])
yellow_cmap_darker = LinearSegmentedColormap.from_list("yellow_cmap_darker", ["#FFFFFF", "#FFFFE0", "#FFDD00", "#CCAA00"])



def load_model(args):
    if args.model.generation_config.do_sample is None:
        del args.model.generation_config.do_sample
    if args.model.generation_config.num_beams is None:
        del args.model.generation_config.num_beams
    if args.model.generation_config.repetition_penalty is None:
        del args.model.generation_config.repetition_penalty
    if args.model.generation_config.no_repeat_ngram_size is None:
        del args.model.generation_config.no_repeat_ngram_size
    if args.model.generation_config.length_penalty is None:
        del args.model.generation_config.length_penalty

    # add 5 extra tokens for the sep_id tokens (e.g., '. ' @ KoCommonGen/ ' 요약: ' @ XL_Sum/ ' 수정: ' @ KoreanGEC)
    args.data.max_length = args.model.generation_config.max_length + args.model.generation_config.max_new_tokens + 5

    if args.model.set_script:
        try: assert args.model.script.max_length % args.data.max_length == 0
        except AssertionError: args.model.script.max_length = args.model.script.max_length - (args.model.script.max_length % args.data.max_length)

    if args.model.hf_model:
        specific_model_type = ""
        if args.model.set_lora:
            specific_model_type += "lora_"
        if args.model.set_script:
            specific_model_type += f"script_{args.model.script.tok_type}_{args.model.script.max_length}t_"
            if args.model.script.do_combination:
                specific_model_type += f"comb-{args.model.script.combination.combination_type}_"
            else:
                specific_model_type += f"red-{args.model.script.reducer}_"
            specific_model_type += f"{args.model.script.fusion}_"

        args.logging.log_dir = os.path.join(f"logs/{args.model.name.replace('/', '_')}/nlg_tasks/{args.data.task_name}/{specific_model_type}{args.model.generation_config.max_length}+{args.model.generation_config.max_new_tokens}t_{args.optim.batch_size}b_{args.optim.grad_acc}s_{args.optim.base_lr}lr_{args.seed}rs")
        args.logging.save_dir = os.path.join(args.logging.log_dir, "ckpt")
        args.logging.tb_dir = os.path.join(args.logging.log_dir, "tb")


    # ----------------------------------------
    #           Set the Trainer
    # ----------------------------------------
    # Get the Tokenizer
    if args.model.hf_model:
        if 'skt/kogpt2' in args.model.name or 'skt/ko-gpt-trinity' in args.model.name:
            tokenizer = AutoTokenizer.from_pretrained(args.model.name,
                                                      bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                      pad_token='<pad>', mask_token='<mask>',
                                                      padding_side='left',
                                                      )
        elif ('EleutherAI/polyglot-ko' in args.model.name) or ('ai-forever/mGPT' in args.model.name):
            tokenizer = AutoTokenizer.from_pretrained(args.model.name)
        elif 'kakaobrain/kogpt' in args.model.name:
            tokenizer = AutoTokenizer.from_pretrained(
                revision='KoGPT6B-ryan1.5b-float16' if args.mixed_precision in ['bf16', 'float16'] else 'KoGPT6B-ryan1.5b',
                bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
            )
        else:
            raise ValueError("It's a Wrong Model Name. Please enter the right model name.")
    else:
        tokenizer = get_gpt2_tokenizer(tok_type=args.data.tok_type,
                                       lang=args.data.language,
                                       max_length=args.data.max_length,
                                       lowercase=True,
                                       clean_text=True,
                                       add_bos_token=False,
                                       bos_token='<|endoftext|>',
                                       eos_token='<|endoftext|>',
                                       unk_token='<unk>',
                                       pad_token='<|endoftext|>',
                                       )
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.custom_tokenizer.config.name in ["jamo_var_info", "bts_units_var_info"]:
            args.model.generation_config.max_length = args.model.generation_config.max_length - (args.model.generation_config.max_length % tokenizer.trunc_num)
            args.model.generation_config.max_new_tokens = args.model.generation_config.max_new_tokens - (args.model.generation_config.max_new_tokens % tokenizer.trunc_num)
            args.data.max_length = args.model.generation_config.max_length + args.model.generation_config.max_new_tokens
            tokenizer.max_length = args.data.max_length

    # Set basic settings for training
    setup_basics(args)

    # Set the Config and Model
    config, model = get_config_and_nlg_model(args, tokenizer)

    args.model.generation_config.pad_token_id = tokenizer.eos_token_id
    args.model.generation_config.bos_token_id = tokenizer.eos_token_id
    args.model.generation_config.eos_token_id = tokenizer.eos_token_id
    if args.model.hf_model:
        for key, value in dict(args.model.generation_config).items():
            setattr(model.generation_config, key, value)
    else:
        generation_config = GenerationConfig.from_pretrained(args.model.name)

        for key, value in dict(args.model.generation_config).items():
            setattr(generation_config, key, value)
        model.generation_config = generation_config

    # Set the Accelerator
    accelerator = Accelerator(
        cpu=(args.device == "cpu"),
        mixed_precision=args.mixed_precision
    )

    trainer = GPTNLGTrainer(args, accelerator, None, tokenizer, model, {'train': None, 'dev': None, 'test': None})
    return trainer


if __name__ == "__main__":
    with initialize(version_base='1.1', config_path="configs/gpt"):
        args = compose(config_name="default",
                       overrides=["mode=nlg_ft",
                                  "task=KoCommonGen",
                                  "model.hf_model=True",
                                  "model.name=skt/kogpt2-base-v2",
                                  "optim.base_lr=1e-02",
                                  "model.set_lora=true",
                                  "model.lora.r=32",
                                  "model.lora.alpha=128",
                                  "model.set_script=true",
                                  "model.script.do_combination=true",
                                  "model.script.fusion=cross_attn",
                                  "model.script.tok_type=jamo_var",
                                  "model.script.max_length=2048",
                                  "logging.log_steps=10000",
                                  "seed=1",
                                  "model.ckpt_dir=/data2/user13/workspace/KOMBO_Generation/logs/skt_kogpt2-base-v2/nlg_tasks/KoCommonGen/lora_script_comb-gru_128+30t_128b_1s_0.005lr_1rs/ckpt/checkpoint-best"
                                  ]
                       )
    trainer = load_model(args)
    subchar_tokenizer = trainer.model.transformer.wte.script_tokenizer
    subword_tokenizer = trainer.tokenizer
    subchar_embeddings = trainer.model.transformer.wte.script_combination.script_embedding.weight.data
    subword_embeddings = trainer.model.transformer.wte.original_layer.weight.data

    save_dir = "analysis/assets"
    os.makedirs(save_dir, exist_ok=True)

    word_lists = [
        ['춥다', '추움', '추위', '추웠어', '춥디춥다'],
        ['걷다', '걷기', '걸어', '걸었어', '걸음'],
        ['돕다', '도움', '도와', '도왔어', '돕기'],
        ['묻다', '물어보다', '물었다', '물어보기', '물어'],
        # ['노랗다', '노란', '노랗게', '노랗기', '노랑'],
    ]
    cmap_list = ['Blues', 'Reds', 'Greens', 'Purples', yellow_cmap_darker]
    # cmap_list = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges']
    for w, word_list in enumerate(word_lists):
        cmap = cmap_list[w]

        # Get each subchar and subword vectors
        subchar_vectors = torch.stack([torch.mean(subchar_embeddings[subchar_tokenizer.encode(word, return_tensors="pt")[0]], dim=0) for word in word_list], dim=0)
        subword_vectors = torch.stack([torch.mean(subword_embeddings[subword_tokenizer.encode(word, return_tensors="pt")[0]], dim=0) for word in word_list], dim=0)

        # Get the fusion vectors
        script = trainer.model.transformer.wte
        inputs = [subword_tokenizer.encode(word, return_tensors="pt")[0].to(trainer.device) for word in word_list]
        fusion_vectors = []
        for i in range(len(inputs)):
            output = script.forward(inputs[i].unsqueeze(0)).squeeze()
            if output.dim() == 1:
                output = output.unsqueeze(0)
            fusion_vectors.append(torch.mean(output, dim=0))
        fusion_vectors = torch.stack(fusion_vectors, dim=0).detach()

        # Plot cosione similarity map between the word embeddings (subchar)
        cos_sim = cosine_similarity(subchar_vectors.cpu().numpy(), subchar_vectors.cpu().numpy())

        tick_font_size = 14
        title_pad = 20
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

        cax = ax1.matshow(cos_sim, cmap=cmap, vmin=0, vmax=1)
        ax1.set_xticklabels([''] + word_list, fontsize=tick_font_size)
        ax1.set_yticklabels([''] + word_list, fontsize=tick_font_size)
        if w == 0:
            ax1.set_title("Subcharacter Embeddings", fontsize=20, pad=title_pad)
        else:
            ax1.set_title(" ", fontsize=20, pad=title_pad)

        # Plot cosione similarity map between the word embeddings (subword)
        cos_sim = cosine_similarity(subword_vectors.cpu().numpy(), subword_vectors.cpu().numpy())
        cax = ax2.matshow(cos_sim, cmap=cmap, vmin=0, vmax=1)
        ax2.set_xticklabels([''] + word_list, fontsize=tick_font_size)
        ax2.set_yticklabels([''] + word_list, fontsize=tick_font_size)
        if w == 0:
            ax2.set_title("Subword Embeddings", fontsize=20, pad=title_pad)
        else:
            ax1.set_title(" ", fontsize=20, pad=title_pad)

        # Plot cosione similarity map between the word embeddings (fusion)
        cos_sim = cosine_similarity(fusion_vectors.cpu().numpy(), fusion_vectors.cpu().numpy())
        cax = ax3.matshow(cos_sim, cmap=cmap, vmin=0, vmax=1)
        ax3.set_xticklabels([''] + word_list, fontsize=tick_font_size)
        ax3.set_yticklabels([''] + word_list, fontsize=tick_font_size)
        if w == 0:
            ax3.set_title("Integrated Subword Embeddings", fontsize=20, pad=title_pad)
        else:
            ax1.set_title(" ", fontsize=20, pad=title_pad)
        fig.colorbar(cax, aspect=10)
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(save_dir, f"embedding_hm_{w}.png"))
