# Low-Rank Few-Shot Adaptation of Vision-Language Models


Here is how to run the experiments:

1. [Installation](#installation) 
2. [Usage](#how-to-execute-CLIP-LoRA) 

A quick guide on how LoRA is implemented in this repository:

3. [LoRA in MultiheadAttention](#lora-in-multiheadattention)

Please consider supporting our work:

4. [Citation](#citation)

If you have any inquiries:

5. [Contact](#contact)
   

## Installation 

### Environment configuration

Our code requires an environment with PyTorch installed. If you don't have one, consider creating a Python environment with:
```bash
conda create -y --name CLIP-LoRA python=3.10.0
conda activate CLIP-LoRA
```
And install Pytorch for instance with:
```bash
pip3 install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2
```

### Datasets installation

Please follow [DATASETS.md](DATASETS.md) to install the datasets.

## How to execute CLIP-LoRA

Execute CLIP-LoRA on the ImageNet dataset with a random seed of 1 by entering the following command:

```bash
python main.py --root_path /path/to/your/data --dataset imagenet --seed 1
```

You can also exectute CLIP-LoRA on the 10 other datasets:

```bash
python main.py --root_path /path/to/your/data --dataset dataset_name --seed 1
```

### Base-to-novel generalization

Use `--setting base2new` to reproduce the class-split protocol used by
CoOp, MMA, and 2SFS. Classes are ordered by their original label, the first
`ceil(C / 2)` classes are used for few-shot training and base evaluation,
and the remaining classes are held out for novel evaluation. Labels are
remapped independently within each split.

```bash
python main.py --root_path /path/to/your/data --dataset dtd --seed 1 \
  --shots 16 --setting base2new --adaptation hydra
```

The run reports zero-shot and adapted base accuracy, novel accuracy, and
their harmonic mean. Base-to-novel checkpoints are stored below a dedicated
`base2new` directory and cannot overwrite standard all-to-all checkpoints.
Use `--setting standard` (the default) for the original CLIP-LoRA protocol.

By default, the repository uses HydraLoRA projections with one shared
low-rank `A` matrix and multiple routed `B` experts. To run the original
CLIP-LoRA architecture, use:

```bash
python main.py --root_path /path/to/your/data --dataset dataset_name --seed 1 --adaptation lora
```

To configure HydraLoRA explicitly:

```bash
python main.py --root_path /path/to/your/data --dataset dataset_name --seed 1 \
  --adaptation hydra --num_experts 4
```

These checkpoints are stored in the `hydra` subdirectory.

To discourage expert collapse and balance router usage:

```bash
python main.py --root_path /path/to/your/data --dataset dataset_name --seed 1 \
  --adaptation hydra --num_experts 4 \
  --hydra_diversity_weight 0.01 --hydra_balance_weight 0.01
```

Both regularization weights default to zero, preserving the clean Hydra
baseline unless explicitly enabled. Regularized checkpoints are stored in the
`hydra_regularized` subdirectory so they cannot overwrite clean Hydra runs.

To gradually sharpen routing during training, use a cosine temperature
schedule. The final temperature is also used for inference:

```bash
python main.py --root_path /path/to/your/data --dataset dataset_name --seed 1 \
  --adaptation hydra --num_experts 4 \
  --hydra_diversity_weight 0.01 --hydra_balance_weight 0.01 \
  --router_temperature 1.0 --router_temperature_end 0.1 \
  --router_temperature_schedule cosine
```

Annealed checkpoints are stored in the `hydra_annealed_regularized`
subdirectory. Use `--router_temperature_schedule fixed` to retain the original
fixed-temperature behavior.

To split one training run across two GPUs with DistributedDataParallel, launch
with `torchrun`. The training batch size is the effective global batch size;
the evaluation batch size is per GPU:

```bash
torchrun --standalone --nproc_per_node=2 main.py \
  --root_path /path/to/your/data --dataset imagenet --seed 1 \
  --batch_size 32 --eval_batch_size 64 \
  --adaptation hydra --num_experts 4 \
  --router_temperature 0.1 \
  --hydra_diversity_weight 0.01 --hydra_balance_weight 0.01 \
  --image_anchor_weight 0.5 --text_anchor_weight 1.0
```

DDP splits the image batch and class prompts across GPUs, then gathers text
features with autograd support before computing the classification loss. Each
GPU still stores a complete CLIP model.

For additional training-memory savings, accumulate smaller microbatches while
preserving the same effective global batch size:

```bash
torchrun --standalone --nproc_per_node=2 main.py \
  --root_path /path/to/your/data --dataset imagenet \
  --batch_size 32 --accumulation_steps 4
```

With two GPUs, this uses a per-GPU microbatch size of `32 / (2 × 4) = 4`,
then performs one optimizer update after four microbatches. The effective
global batch remains `4 × 2 × 4 = 32`. Learning-rate scheduling,
`--n_iters`, logging intervals, and validation intervals remain measured in
optimizer steps. `--batch_size` must be divisible by
`world_size × accumulation_steps`.

Training runs with a `--save_path` also create `training_log.jsonl` beside the
checkpoint. It records configuration, periodic training metrics, validation
results, and final test accuracy. Adjust the recording frequency with
`--log_interval`; optionally record a validation curve with `--val_interval`:

```bash
python main.py --root_path /path/to/your/data --dataset dataset_name \
  --save_path /your/save/path --log_interval 10 --val_interval 500
```

The step records include classification and regularization losses, accuracy,
learning rate, router temperature, gradient and adapter norms, expert cosine
similarity, router usage, AMP scale, and skipped
optimizer updates. Periodic validation is disabled by default because it adds
evaluation cost.

You can optionally provide a save_path to save the LoRA modules, which can be reload easily with the --eval_only argument. The code will automatically check if your trained LoRA with the corresponding rank, alpha, encoder, params and position to ensure compatibility. The folder will be structured like that:
```
/your/save/path
└── backbone
    └── dataset
        └── Xshots
            ├── seedY
```

Here is the command line:
```bash
python main.py --root_path /path/to/your/data --dataset dataset_name --seed 1 --save_path /your/save/path --eval_only 
```

## LoRA in MultiheadAttention

The `PlainMultiheadAttentionLoRA` class in `loralib/layers.py` extends the standard PyTorch multi-head attention mechanism by incorporating Low-Rank Adaptation (LoRA). This class constructs explicit linear modules for each component of the attention mechanism—query (`q`), key (`k`), value (`v`), and output (`o`)—providing a structured and adaptable foundation for your experiments.

### Class Overview

`PlainMultiheadAttentionLoRA` takes an existing `nn.MultiheadAttention` module, replicates its configuration, and integrates LoRA linear modules.

### Key Features

- **Parameter Initialization:** The initialization process involves copying weights and biases from a pre-existing multi-head attention model. Each LoRA module (`q`, `k`, `v`, `o`) is adapted based on the specified requirements in the `enable_lora` list.
- **LoRA Integration:** The replacement of standard linear layers with `LinearLoRA` layers introduces low-rank matrices, which are parameterized by the rank of adaptation (`r`) and the scaling factor (`lora_alpha`).
- **Forward Pass:** The `forward_module` method manages the attention computation, incorporating optional dropout settings on the LoRA modules.
