import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"

import json
import torch
from argparse import Namespace
from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from hydra import compose, initialize
from accelerate import Accelerator
from utils.gen_utils import setup_basics
from transformers import AutoTokenizer, AutoModelForCausalLM

rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(os.getcwd())
# from KOMBO.pretraining.srcs.functions import init_random
from KOMBO.pretraining.utils.base_parser import ArgsBase
from KOMBO.pretraining.utils.logger import get_logger
from KOMBO.nlu_tasks.srcs.nlu_trainer import Trainer as NLU_Trainer
from nlg_tasks.srcs.trainer import GPTNLGTrainer
from nlg_tasks.scripts.run_gpt_finetuning import get_config_and_nlg_model


def load_model(args):
    if args.model.set_script:
        try: assert args.model.script.max_length % args.data.max_length == 0
        except AssertionError: args.model.script.max_length = args.model.script.max_length - (args.model.script.max_length % args.data.max_length)

    # ----------------------------------------
    #           Set the Trainer
    # ----------------------------------------
    # Get the Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model.name, padding_side="right")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set basic settings for training
    setup_basics(args)

    # Set the Config and Model
    config, model = get_config_and_nlg_model(args, tokenizer)

    # Set the Accelerator
    accelerator = Accelerator(
        cpu=(args.device == "cpu"),
        mixed_precision=args.mixed_precision
    )

    trainer = GPTNLGTrainer(args, accelerator, None, tokenizer, model, {'train': None, 'dev': None, 'test': None})
    return trainer


################################################
#        Our Model + BERT-base Settings
################################################
# parser = ArgsBase().add_nlu_task_args()
# temp_args = parser.parse_args()
# temp = vars(temp_args)
# config = json.load(open(f"KOMBO/nlu_tasks/data_configs/{temp['task_name']}/config.json"))
# for arg in config:
#     temp[arg] = config[arg]
# args = Namespace(**temp)
#
#
# args.tok_type = 'morphemeSubword'
# args.tok_vocab_size = '32k'
# args.tok_name = 'morphemeSubword_ko_wiki_32k'
# args.save_dir = 'KOMBO/logs/bert-base/morphemeSubword_ko_wiki_32k/pretraining/128t_128b_1s_5e-05lr_42rs/ckpt'
# args.model_name = 'bert-base'
# args.set_script = True
# args.load_script = True
# args.set_lora = False
# args.script_tok_type = 'jamo_distinct'
# args.script_tok_vocab_size = '200'
# args.script_tok_name = 'jamo_distinct_ko_200'
# args.script_intermediate_size = 768
# args.script_num_attention_heads = 2
#
#
# logger = get_logger()
# trainer = NLU_Trainer(args, logger)
#
# subword_tokenizer = trainer.tokenizer
# subcharacter_tokenizer = trainer.model.bert.embeddings.word_embeddings.script_tokenizer
#
# subcharacter_embedding = trainer.model.bert.embeddings.word_embeddings.script_combination.script_embedding
# script_subword_embedding = trainer.model.bert.embeddings.word_embeddings.script_combination
# original_subword_embedding = trainer.model.bert.embeddings.word_embeddings.original_layer
#
# final_subword_embedding = trainer.model.bert.embeddings.word_embeddings
# final_subword_embedding_w_position = trainer.model.bert.embeddings


################################################
#          GPT-2-base + Ours Settings
################################################
with initialize(version_base='1.1', config_path="configs/gpt"):
    args = compose(config_name="default",
                   overrides=["mode=nlg_ft",
                              "task=KorSTS",
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
                              "model.script.combination.intermediate_size=768",
                              "logging.log_steps=10000",
                              "seed=1",
                              "model.ckpt_dir=/data2/user13/workspace/script/logs/skt_kogpt2-base-v2/nlu_tasks/KorSTS/lora_script_jamo_var_2048t_comb-gru_768idim_cross_attn_256t_8b_1s_5ep_5e-05lr_42rs_bf16/ckpt/checkpoint-2157"
                              ]
                   )

trainer = load_model(args)
script_kogpt2_tokenizer = trainer.model.base_model.wte.script_tokenizer
script_kogpt2_embedding = trainer.model.base_model.wte.script_combination

del trainer


################################################
#          KoGPT3 + Ours Settings
################################################
with initialize(version_base='1.1', config_path="configs/gpt"):
    args = compose(config_name="default",
                   overrides=["mode=nlg_ft",
                              "task=KorSTS",
                              "model.hf_model=True",
                              "model.name=skt/ko-gpt-trinity-1.2B-v0.5",
                              "optim.base_lr=1e-02",
                              "model.set_lora=true",
                              "model.lora.r=32",
                              "model.lora.alpha=128",
                              "model.set_script=true",
                              "model.script.do_combination=true",
                              "model.script.fusion=cross_attn",
                              "model.script.tok_type=jamo_var",
                              "model.script.max_length=2048",
                              "model.script.combination.intermediate_size=768",
                              "logging.log_steps=10000",
                              "seed=1",
                              "model.ckpt_dir=/data2/user13/workspace/script/logs/skt_ko-gpt-trinity-1.2B-v0.5/nlu_tasks/KorSTS/lora_script_jamo_var_2048t_comb-gru_768idim_cross_attn_256t_8b_1s_5ep_3e-05lr_42rs_bf16/ckpt/checkpoint-2157"
                              ]
                   )

trainer = load_model(args)
script_kogpt3_tokenizer = trainer.model.base_model.wte.script_tokenizer
script_kogpt3_embedding = trainer.model.base_model.wte.script_combination

del trainer


################################################
#        KOMBO-base Settings
################################################
parser = ArgsBase().add_nlu_task_args()
temp_args = parser.parse_args()
temp = vars(temp_args)
config = json.load(open(f"KOMBO/nlu_tasks/data_configs/{temp['task_name']}/config.json"))
for arg in config:
    temp[arg] = config[arg]
args = Namespace(**temp)


args.tok_type = 'jamo_distinct'
args.tok_vocab_size = '200'
args.tok_name = 'jamo_distinct_ko_200'
args.save_dir = 'KOMBO/logs/kombo-base/jamo_distinct_ko_200/pretraining/span-character-mlm_jamo-trans3_gru_conv1-cjf_repeat_gru-up-res_128t_128b_1s_5e-05lr_42rs/ckpt'
args.model_name = 'kombo-base'
args.set_script = False
args.load_script = False
args.set_lora = False


logger = get_logger()
trainer = NLU_Trainer(args, logger)

kombo_tokenizer = trainer.tokenizer
kombo_embedding = trainer.model.bert.embeddings.token_embedding


################################################
#              KoGPT2-base Settings
################################################
kogpt2_tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
kogpt2_model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2", trust_remote_code=True, use_safetensors=True)
kogpt2_embedding = kogpt2_model.base_model.wte


################################################
#              KoGPT3-1.2B Settings
################################################
kogpt3_tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
kogpt3_model = AutoModelForCausalLM.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5", trust_remote_code=True, use_safetensors=True)
kogpt3_embedding = kogpt3_model.base_model.wte


################################################
#               mGPT-1.3B Settings
################################################
mgpt_tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT")
mgpt_model = AutoModelForCausalLM.from_pretrained("ai-forever/mGPT", trust_remote_code=True, use_safetensors=True)
mgpt_embedding = mgpt_model.base_model.wte


################################################
#           EXAONE-3.5-7.8B Settings
################################################
model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
exaone_tokenizer = AutoTokenizer.from_pretrained(model_name)
exaone = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
exaone_embedding = exaone.base_model.wte


################################################
#              EEVE-10.8B Settings
################################################
eeve_tokenizer = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-10.8B-v1.0")
eeve_model = AutoModelForCausalLM.from_pretrained("yanolja/EEVE-Korean-10.8B-v1.0")
eeve_embedding = eeve_model.base_model.embed_tokens


################################################
#                Qwen-14B Settings
################################################
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B")
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B")
qwen_embedding = qwen_model.base_model.embed_tokens




################################################
#            Word Vector Projection
################################################
# def get_our_bert_embedding_instance(word_variations):
#     subword_tokens = []
#     for variation in word_variations:
#         tokens = [tok for tok in subword_tokenizer.encode(variation) if tok not in subword_tokenizer.special_tokens_encoder.values()]
#         subword_tokens.append(torch.LongTensor(tokens).to(trainer.device))
#     original_vectors = [torch.sum(original_subword_embedding(tokens), dim=0)for tokens in subword_tokens]
#
#     subcharacter_tokens = []
#     for variation in word_variations:
#         tokens = [subcharacter_tokenizer.cls_token_id]
#         tokens += [tok for tok in subcharacter_tokenizer.encode(variation) if tok not in subcharacter_tokenizer.special_tokens_encoder.values()]
#         subcharacter_tokens.append(torch.LongTensor(tokens).to(trainer.device))
#     # subcharacter_vectors = [torch.sum(subcharacter_embedding(tokens), dim=0) for tokens in subcharacter_tokens]
#     script_vectors = [torch.sum(script_subword_embedding(tokens.reshape(1, -1), text_input=[word_variations[i]]).squeeze()[1: len(tokens), ], dim=0) for i, tokens in enumerate(subcharacter_tokens)]
#     # final_subword_vectors = [torch.sum(final_subword_embedding(tokens.reshape(1, -1)).squeeze()[1: len(tokens), ], dim=0)for tokens in subcharacter_tokens]
#
#     # return original_vectors, subcharacter_vectors, script_vectors, final_subword_vectors
#     return original_vectors, script_vectors


def get_our_gpt_embedding_instance(word_variations, tokenizer, embedding):
    subcharacter_tokens = []
    for variation in word_variations:
        tokens = [tok for tok in tokenizer.encode(variation) if tok not in tokenizer.special_tokens_encoder.values()]
        subcharacter_tokens.append(torch.LongTensor(tokens).to(embedding.script_embedding.weight.device))
    script_vectors = [torch.sum(embedding(tokens.reshape(1, -1), text_input=[word_variations[i]]).squeeze()[1: len(tokens), ], dim=0) for i, tokens in enumerate(subcharacter_tokens)]

    return script_vectors


def get_baseline_embedding_instance(word_variations, tokenizer, embedding):
    subword_tokens = []
    for variation in word_variations:
        tokens = [tok for tok in tokenizer.encode(variation)]
        subword_tokens.append(torch.LongTensor(tokens).to(embedding.weight.device))

    subword_vectors = [torch.sum(embedding(tokens), dim=0) for tokens in subword_tokens]
    return subword_vectors



################################################
#                Plot Function
################################################

def plot_reduced_vectors(embeddings, title, reduce_method="pca", legend=False, text_label=True, text_margin=0.001, save_fig=False, save_file_name=None):
    plt.figure(figsize=(16, 8))
    for embedding in embeddings:
        vectors = embedding['vectors']
        label = embedding['label']
        color = embedding['color']
        markers = embedding['markers']

        if reduce_method == "pca":
            reduced_vectors = PCA(n_components=2).fit_transform(torch.stack(vectors).detach().to(device='cpu', dtype=torch.float32).cpu().numpy())
            reduced_vectors = np.sqrt(np.abs(reduced_vectors)) * np.sign(reduced_vectors)
        elif reduce_method == "tsne":
            reduced_vectors = TSNE(n_components=2, perplexity=2, n_iter=1000).fit_transform(torch.stack(vectors).detach().to(device='cpu', dtype=torch.float32).cpu().numpy())
            reduced_vectors = reduced_vectors / np.linalg.norm(reduced_vectors, axis=1).reshape(-1, 1)
        else:
            raise NotImplementedError("Please specify a reduction method among pca, tsne.")

        for i in range(len(reduced_vectors)):
            plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1], c=color, marker=markers[i], s=100,
                        label=f"{label}" if i == 0 else "")
            if text_label:
                plt.text(reduced_vectors[i, 0] + text_margin, reduced_vectors[i, 1] + text_margin, word_variations[i], fontsize=9)

    if title is not None:
        plt.title(title)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if legend:
        plt.legend(loc="upper right")

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    if save_fig:
        save_dir = "analysis/assets"
        os.makedirs(save_dir, exist_ok=True)
        if save_file_name is None:
            plt.savefig(os.path.join(save_dir, f"{title}.png"), transparent=True, dpi=600, format="png")
        else:
            plt.savefig(os.path.join(save_dir, f"{save_file_name}.png"), transparent=True, dpi=600, format="png")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    word_variations = [
        "춥다", "추웠다", "추움", "춥습니다", "추워죽겠다",
    ]

    # original_subword_vectors, script_subword_vectors = get_our_bert_embedding_instance(word_variations)
    script_kogpt2_subword_vectors = get_our_gpt_embedding_instance(word_variations, script_kogpt2_tokenizer, script_kogpt2_embedding)
    script_kogpt3_subword_vectors = get_our_gpt_embedding_instance(word_variations, script_kogpt3_tokenizer, script_kogpt3_embedding)

    kombo_subword_vectors = get_baseline_embedding_instance(word_variations, kombo_tokenizer, kombo_embedding)
    kogpt2_subword_vectors = get_baseline_embedding_instance(word_variations, kogpt2_tokenizer, kogpt2_embedding)
    kogpt3_subword_vectors = get_baseline_embedding_instance(word_variations, kogpt3_tokenizer, kogpt3_embedding)
    # mgpt_subword_vectors = get_baseline_embedding_instance(word_variations, mgpt_tokenizer, mgpt_embedding)
    exaone_subword_vectors = get_baseline_embedding_instance(word_variations, exaone_tokenizer, exaone_embedding)
    # eeve_subword_vectors = get_baseline_embedding_instance(word_variations, eeve_tokenizer, eeve_embedding)
    # qwen_subword_vectors = get_baseline_embedding_instance(word_variations, qwen_tokenizer, qwen_embedding)

    markers = ['o', 's', '^', 'D', 'v']  # Define markers for each variation
    embedding_instances = [
        # {"vectors": original_subword_vectors, "label": r'$\text{BERT}_{base}$ Subword', "color": "salmon", "markers": markers},
        {"vectors": kombo_subword_vectors, "label": r'$\text{KOMBO}_{base}$ Subword', "color": "black",
         "markers": markers},
        {"vectors": kogpt2_subword_vectors, "label": r'$\text{KoGPT2}_{base}$ Subword', "color": "orange",
         "markers": markers},
        {"vectors": kogpt3_subword_vectors, "label": r'KoGPT3-1.2B Subword', "color": "green", "markers": markers},
        # {"vectors": mgpt_subword_vectors, "label": r'mGPT-1.3B Subword', "color": "purple", "markers": markers},
        {"vectors": exaone_subword_vectors, "label": r'EXAONE-3.5-7.8B Subword', "color": "blue", "markers": markers},
        # {"vectors": eeve_subword_vectors, "label": r'EEVE-10.8B Subword', "color": "orange", "markers": markers},
        # {"vectors": qwen_subword_vectors, "label": r'Qwen-2.5-14B Subword', "color": "purple", "markers": markers},
        {"vectors": script_kogpt2_subword_vectors, "label": r'$\text{KoGPT2}_{base}$ + SCRIPT', "color": "red",
         "markers": markers},
        # {"vectors": script_kogpt3_subword_vectors, "label": r'$\text{KoGPT3}_{base}$ + SCRIPT', "color": "teal", "markers": markers},
    ]

    plot_reduced_vectors(embeddings=embedding_instances,
                         reduce_method="pca",
                         title=None,
                         legend=False,
                         text_label=False,
                         text_margin=0.03,
                         save_fig=True,
                         save_file_name="word_variation"
                         )

