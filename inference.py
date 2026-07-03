import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer


# ---------- output setup ----------
PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_MD = PROJECT_ROOT / "result.md"
IMG_DIR = PROJECT_ROOT / "results_img"
if IMG_DIR.exists():
    shutil.rmtree(IMG_DIR)
IMG_DIR.mkdir(parents=True, exist_ok=True)

_md_lines: list[str] = []


def md_write(line: str = "") -> None:
    print(line)
    _md_lines.append(line)


def md_flush() -> None:
    RESULT_MD.write_text("\n".join(_md_lines) + "\n", encoding="utf-8")


def save_image(img: Image.Image, name: str) -> str:
    path = IMG_DIR / name
    img.save(path)
    return f"results_img/{name}"


# ---------- model init ----------
# model_path = "/workspace/lora_merged/bagel_8K/"
model_path = "/workspace/BAGEL-7B-MoT"

llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config,
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act="gelu_pytorch_tanh",
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)


# ---------- multi-GPU dispatch ----------
max_mem_per_gpu = "40GiB"

device_map = infer_auto_device_map(
    model,
    max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)
print(device_map)

same_device_modules = [
    "language_model.model.embed_tokens",
    "time_embedder",
    "latent_pos_embed",
    "vae2llm",
    "llm2vae",
    "connector",
    "vit_pos_embed",
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        device_map[k] = first_device if k in device_map else "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(model_path, "ema.safetensors"),
    device_map=device_map,
    offload_buffers=True,
    dtype=torch.bfloat16,
    force_hooks=True,
    offload_folder="/tmp/offload",
)
model = model.eval()
print("Model loaded")


# ---------- inferencer + seed ----------
inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids,
)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------- hyperparameters ----------
inference_hyper = dict(
    max_think_token_n=4096,
    do_sample=True,
    text_temperature=0.3,
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
)


# ---------- task runner ----------
def run_task(section: str, slug: str, image_path: str, prompt: str) -> None:
    md_write(f"## {section}\n")

    image = Image.open(image_path)
    input_rel = save_image(image, f"{slug}_input.png")
    md_write(f"**Input image:**\n")
    md_write(f"![{slug} input]({input_rel})\n")
    md_write(f"**Prompt:**\n")
    md_write("```")
    md_write(prompt)
    md_write("```\n")

    output_list = inferencer(
        image=image, text=prompt, understanding_output=False, think=True, **inference_hyper
    )

    md_write("**Output:**\n")
    text_round = 0
    img_round = 0
    for out_item in output_list:
        if isinstance(out_item, str):
            md_write(f"*Round {text_round}:*\n")
            md_write("```")
            md_write(out_item)
            md_write("```\n")
            text_round += 1
        elif isinstance(out_item, Image.Image):
            rel = save_image(out_item, f"{slug}_out_{img_round}.png")
            md_write(f"![{slug} output {img_round}]({rel})\n")
            img_round += 1
    md_write("---\n")
    md_flush()


md_write("# ThinkMorph Inference Results\n")
md_flush()

run_task(
    section="Jigsaw Assembly",
    slug="jigsaw",
    image_path="test_images/Jigsaw_Assembly.jpg",
    prompt=(
        "The image below is divided into four parts by white strips, forming a 2×2 jigsaw puzzle. "
        "The parts are labeled \"1\" (top-left), \"2\" (top-right), \"3\" (bottom-left), and "
        "\"4\" (bottom-right). These parts are from a single original image but have been shuffled. "
        "Your task is to determine the correct arrangement of the physically labeled parts to "
        "reconstruct the natural image.\n\nSelect the correct statement from the following choices:\n\n"
        "(A) The top-left part should be Part 1; the top-right part should be Part 2; the bottom-left part should be Part 4; and the bottom-right part should be Part 3.\n"
        "(B) The top-left part should be Part 1; the top-right part should be Part 3; the bottom-left part should be Part 4; and the bottom-right part should be Part 2.\n"
        "(C) The top-left part should be Part 3; the top-right part should be Part 2; the bottom-left part should be Part 1; and the bottom-right part should be Part 4.\n"
        "(D) The top-left part should be Part 2; the top-right part should be Part 4; the bottom-left part should be Part 1; and the bottom-right part should be Part 3."
    ),
)

run_task(
    section="Visual Search",
    slug="visual_search",
    image_path="test_images/Visual_Search.jpg",
    prompt="What is the color of the cart?\nA: red\nB: white\nC: black\nD: green",
)

run_task(
    section="Spatial Navigation",
    slug="spatial_navigation",
    image_path="test_images/Spatial_Navigation.jpg",
    prompt=(
        "You are a maze solver. Your goal is to guide a player from the start to the goal on a "
        "grid map while avoiding holes. The player can move one square at a time in the directions "
        "left (L), right (R), up (U), or down (D). The frozen lake is not slippery; the player will "
        "always move in the intended direction. Moving off the edge or falling into a hole results "
        "in failure. Reaching the goal means success. Provide your solution as a sequence of moves "
        "wrapped in \\boxed{{}}, such as \\boxed{L,R,U,D}. The moves should be comma-separated."
    ),
)

print(f"\nWrote {RESULT_MD}")
print(f"Images in {IMG_DIR}")
