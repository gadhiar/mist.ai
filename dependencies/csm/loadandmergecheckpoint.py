import os
import re
import torch
from models import Model
from safetensors.torch import save_file, load_file 

from lora import (
    remove_lora_modules,
    merge_lora_weights,
    strip_bias_keys,
    DEVICE,
    OUTPUT_DIR,
    replace_linear_with_lora,
)
MODEL_NAME = "sesame/csm-1b"
R=32
APLHA=32

def find_latest_checkpoint(dir_path):
    checkpoints = [
        (int(re.search(r"checkpoint-epoch-(\d+)", d).group(1)), os.path.join(dir_path, d))
        for d in os.listdir(dir_path)
        if os.path.isdir(os.path.join(dir_path, d)) and "checkpoint-epoch" in d
    ]
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found.")
    latest_epoch, latest_path = max(checkpoints, key=lambda x: x[0])
    print(f"Latest checkpoint: epoch {latest_epoch} -> {latest_path}")
    return latest_path

def load_checkpoint_and_merge():
    print("Loading base model...")
    model = Model.from_pretrained(MODEL_NAME).to(DEVICE)

    print("Applying LoRA structure to the model...")
    target_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    model = replace_linear_with_lora(model, r=R, alpha=APLHA, dropout=0.0, target_linear_names = target_layers)
    checkpoint_path = find_latest_checkpoint(OUTPUT_DIR)
    
    print(f"Loading state dictionary from safetensors file...")
    state_dict = load_file(os.path.join(checkpoint_path, "model.safetensors"), device=DEVICE)

    print("Loading weights into the model...")
    model.load_state_dict(state_dict, strict=False)

    print("Merging LoRA weights into base model...")
    merge_lora_weights(model)

    print("Replacing LoRALinear modules with standard nn.Linear...")
    model = remove_lora_modules(model)

    print("Stripping bias keys for final clean model...")
    merged_state = strip_bias_keys(model.state_dict())

    final_path = os.path.join(OUTPUT_DIR, "model.safetensors")
    save_file(merged_state, final_path)
    print(f"Merged and cleaned model saved to: {final_path}")

if __name__ == "__main__":
    load_checkpoint_and_merge()
