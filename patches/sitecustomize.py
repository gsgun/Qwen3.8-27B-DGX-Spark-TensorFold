import os, datetime
import torch.distributed.constants
import torch.distributed.distributed_c10d

# Clamp default NCCL timeout from 600s down to 60s to prevent 10-minute 100% CPU lockups
timeout_sec = int(os.environ.get('TORCH_NCCL_COLLECTIVE_TIMEOUT_SEC', 60))
torch.distributed.constants.default_pg_nccl_timeout = datetime.timedelta(seconds=timeout_sec)
torch.distributed.distributed_c10d.default_pg_nccl_timeout = datetime.timedelta(seconds=timeout_sec)
