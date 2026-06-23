import functools
import pickle


try:
    import torch
    import torch.distributed as dist
except ImportError:
    torch = None
    dist = None


def get_dist_info():
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)
        return None

    return wrapper


def get_world_size():
    return get_dist_info()[1]


def get_rank():
    return get_dist_info()[0]


def is_main_process():
    return get_rank() == 0


def synchronize():
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return
    if dist.get_world_size() > 1:
        dist.barrier()


def all_gather(data):
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    if torch is None:
        raise RuntimeError("torch is required for distributed all_gather")

    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    local_size = torch.IntTensor([tensor.numel()]).to("cuda")
    size_list = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = [torch.ByteTensor(size=(max_size,)).to("cuda") for _ in size_list]
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor_item in zip(size_list, tensor_list):
        buffer = tensor_item.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    return data_list
