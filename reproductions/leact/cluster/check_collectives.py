"""Check the allocated node's NCCL collective before full-parameter SFT."""

from datetime import timedelta
import os

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        size = dist.get_world_size()
        value = torch.tensor([rank + 1.0], device="cuda")
        dist.all_reduce(value)
        expected = size * (size + 1) / 2
        if value.item() != expected:
            raise RuntimeError(f"NCCL returned {value.item()}, expected {expected}")
        dist.barrier()
        if rank == 0:
            print(f"NCCL all-reduce passed on {size} GPUs", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
