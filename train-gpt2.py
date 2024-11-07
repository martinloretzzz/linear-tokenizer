# sudo apt-get install unzip
# pip install transformers wandb tiktoken
# unzip /workspace/dataset-lin.zip -d /workspace/dataset/
# unzip /workspace/dataset-ref.zip -d /workspace/dataset/
# torchrun --standalone --nproc_per_node=2 train-gpt2.py
# python train-gpt2.py

import json
import math
import os
import random
import time

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F

from hellaswag import get_most_likely_row, iterate_examples, render_example
from model import GPT, get_model_config
from tokenizer import LinearTokenizer

ENABLE_WANDB = True
LINEAR_TOKENIZER = False

data_root = "dataset/content/data/"

total_batch_size = 524288 # 262144 # 524288 # 2**19, ~0.5M, in number of tokens
B = 4 # 64
T = 1024 # sequence length

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # 715
max_steps = 19073 # 19073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

model_type = "gpt2"
load_pretrained = False
checkpoint_path = None

tokenizer_name = 'lin' if LINEAR_TOKENIZER else 'bpe'
project_name = f"{tokenizer_name}{'-full' if max_steps >= 10000 else ''}"


if LINEAR_TOKENIZER:
    with open('./vocab-gpt2-tt.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    tokenizer = LinearTokenizer(vocab, space_expand_vocab=False)
else:
    tokenizer = tiktoken.get_encoding("gpt2")


with open('wandb.txt', 'r') as file:
    wandb_key = file.read()


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train-gpt2.py

import torch.distributed as dist
# run the training loop
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

config = get_model_config(model_type, vocab_size=50304)
# create model
model = GPT(config)
if load_pretrained:
    model = GPT.from_pretrained(model_type) # or init from OpenAI GPT-2

start_step = 0
if checkpoint_path is not None:
    loaded = torch.load(checkpoint_path, weights_only=False)
    start_step = loaded["step"]
    model_checkpoint = loaded["model"]
    model.load_state_dict(model_checkpoint)
    print(f"Load model from {checkpoint_path}. Continue at step {start_step}.")

model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# Test if model throws any errors
x, y = val_loader.next_batch()
print(x.shape, y.shape)

x, y = train_loader.next_batch()
x, y = x.to(device), y.to(device)
with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
    logits, loss, acc = model(x, y)
print(acc)


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type, master_process=master_process)

if master_process:
    wandb.login(key=wandb_key)
    model_artifact_name = f"{project_name}-model-{random.randint(0, 1000)}"
    run = wandb.init(
        project="linear-tokenizer",
        name=project_name,
        config={
            "tokenizer": tokenizer_name,
            "model_type": model_type,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "total_batch_size": total_batch_size,
            "B": B,
            "T": T
        },
        mode=None if ENABLE_WANDB else "disabled"
    )

for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 25 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum, val_acc_accum = 0.0, 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss, acc = model(x, y)
    
                val_loss_accum += (loss / val_loss_steps).detach()
                val_acc_accum += (acc / val_loss_steps).detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_acc_accum, op=dist.ReduceOp.AVG)

        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f} | acc: {val_acc_accum.item():.2f}")
            run.log({"step": step, "val/loss": val_loss_accum.item(), "val/acc": val_acc_accum.item()})

            if step > 0 and (step % 4000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = f"model_{step:05d}.pt"
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

                artifact = wandb.Artifact(name=model_artifact_name, type="model")
                artifact.add_file(local_path=checkpoint_path)
                run.log_artifact(artifact)


    # once in a while evaluate hellaswag
    if (step % 100 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example, tokenizer)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _, __ = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            run.log({"step": step, "val/hellaswag_acc": acc_norm, "val/hellaswag_correct": num_correct_norm})


    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 100 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = tokenizer.encode("Hello, I'm a language model,")

        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _, __ = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :]# (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = tokenizer.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum, acc_accum = 0.0, 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss, acc = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        acc_accum += acc / grad_accum_steps
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        dist.all_reduce(acc_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        if step % 10 == 0:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | acc: {acc_accum.item():.6f}  | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        run.log({"step": step, "loss": loss_accum.item(), "acc": acc_accum.item(), "lr":lr, "norm":norm, "dt":1000*dt, "tokens_per_sec": tokens_per_sec })

if master_process:
    run.finish()

if ddp:
    destroy_process_group()
