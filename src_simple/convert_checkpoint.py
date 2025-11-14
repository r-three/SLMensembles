#!/usr/bin/env python3
import os
import argparse
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from transformers import AutoConfig, AutoModelForCausalLM
from simple_checkpoint import AppState

def maybe_init_dist():
    if dist.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            # Set CUDA device before initializing process group
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            dist.init_process_group(backend=os.environ.get("DIST_BACKEND", "nccl"),
                                    timeout=timedelta(minutes=30))
        return True, dist.get_rank(), dist.get_world_size()
    return False, 0, 1

def main():
    p = argparse.ArgumentParser("DCP dir → single .pt")
    p.add_argument("--checkpoint-dir", required=True, help="DCP directory (has __*_*.distcp)")
    p.add_argument("--student-model-name", required=True, help="HF id or local config dir (architecture only)")
    p.add_argument("--out", required=True, help="Output .pt file path")
    p.add_argument("--owner-rank", type=int, default=0, help="Rank that performs load+save (default 0)")
    args = p.parse_args()

    is_dist, rank, _ = maybe_init_dist()
    owner = (rank == args.owner_rank)

    # Build the SAME architecture (no weights yet)
    cfg = AutoConfig.from_pretrained(args.student_model_name)
    model = AutoModelForCausalLM.from_config(cfg).to("cpu")

    # Dummy optimizer just to satisfy AppState schema
    dummy_optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 1-rank subgroup so only the owner participates in the collective
    pg = dist.new_group(ranks=[args.owner_rank]) if is_dist else None

    if owner:
        # Load with the SAME top-level key used when saving: {"app": AppState(...)}
        app = AppState(model=model, optimizer=dummy_optim, lr_scheduler=None)
        dcp.load(state_dict={"app": app},
                 checkpoint_id=args.checkpoint_dir,
                 process_group=pg)

        # Save a plain state_dict for easy inference loads
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({"model_state_dict": cpu_state}, args.out)
        print(f"[Rank {rank}] wrote {args.out}")

    if is_dist:
        # Set device before barrier to avoid NCCL warning
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            dist.barrier(device_ids=[device])
        else:
            dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
