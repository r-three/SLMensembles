#!/usr/bin/env python3
import os
import argparse
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from transformers import AutoConfig, AutoModelForCausalLM


def maybe_init_dist():
    if dist.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend=os.environ.get("DIST_BACKEND", "nccl"),
                                    timeout=timedelta(minutes=30))
        return True, dist.get_rank(), dist.get_world_size()
    return False, 0, 1


def main():
    p = argparse.ArgumentParser("DCP dir → single .pt")
    p.add_argument("--checkpoint-dir", required=True, help="DCP directory (contains __*_*.distcp)")
    p.add_argument("--student-model-name", required=True, help="HF id or local cfg dir (for architecture only)")
    p.add_argument("--out", required=True, help="Output .pt path")
    p.add_argument("--owner-rank", type=int, default=0, help="Rank that actually loads+saves (default 0)")
    args = p.parse_args()

    is_dist, rank, _ = maybe_init_dist()
    owner = (rank == args.owner_rank)

    # build the architecture only (no weights)
    cfg = AutoConfig.from_pretrained(args.student_model_name)
    model = AutoModelForCausalLM.from_config(cfg).to("cpu")

    # tiny optimizer just to satisfy typical DCP structures
    dummy_optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 1-rank subgroup so dcp.load is collective only for the owner
    pg = dist.new_group(ranks=[args.owner_rank]) if is_dist else None
    if owner:
        dcp.load(
            state_dict={"model": model, "optimizer": dummy_optim},
            checkpoint_id=args.checkpoint_dir,
            process_group=pg
        )
        # save a plain state_dict for easy inference loads
        os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
        torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, args.out)
        print(f"[Rank {rank}] wrote {args.out}")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
