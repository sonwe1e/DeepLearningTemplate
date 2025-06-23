import torch
import numpy as np
import queue as Queue
import threading
import torch.utils.data as data
from concurrent.futures import ThreadPoolExecutor
from configs.option import get_option
from .augments import train_transform, valid_transform


class PrefetchGenerator(threading.Thread):
    """A general prefetch generator.

    Reference: https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

    Args:
        generator: Python generator.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, generator, num_prefetch_queue):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(num_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(data.DataLoader):
    """Prefetch version of dataloader.

    Reference: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#

    Args:
        num_prefetch_queue (int): Number of prefetch queue.
        kwargs (dict): Other arguments for dataloader.
    """

    def __init__(self, num_prefetch_queue, **kwargs):
        self.num_prefetch_queue = num_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_prefetch_queue)


class CPUPrefetcher:
    """CPU prefetcher.

    Args:
        loader: Dataloader.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)


class CUDAPrefetcher:
    """CUDA prefetcher.

    Reference: https://github.com/NVIDIA/apex/issues/304#

    It may consume more GPU memory.

    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader, opt):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device(
            f"cuda:{opt.devices[0]}" if len(opt.devices) != 0 else "cpu"
        )
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)  # self.batch is a dict
        except StopIteration:
            self.batch = None
            return None
        # put tensors to gpu
        with torch.cuda.stream(self.stream):
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    self.batch[k] = self.batch[k].to(
                        device=self.device, non_blocking=True
                    )

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()


# 新增：可枚举的预取器迭代器类
class PrefetcherIterator:
    """可迭代的预取器包装类，使预取器可以在for循环中使用enumerate。

    Args:
        prefetcher: CPU或CUDA预取器。
    """

    def __init__(self, prefetcher, length=None):
        self.prefetcher = prefetcher
        self.length = length
        self.iterator = None

    def __iter__(self):
        self.prefetcher.reset()
        batch = self.prefetcher.next()

        iterator = range(self.length) if self.length is not None else iter(int, 1)

        for i in iterator:
            if batch is None:
                break
            yield batch
            batch = self.prefetcher.next()

    def __len__(self):
        return self.length if self.length is not None else 0


class Dataset(torch.utils.data.Dataset):
    def __init__(self, phase, opt, train_transform=None, valid_transform=None):
        self.phase = phase
        self.data_path = opt.data_path
        self.transform = train_transform if phase == "train" else valid_transform
        self.image_list = [0] * 100

    def __getitem__(self, index):
        image = np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8)
        label = 1
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return {"image": image, "label": label}

    def __len__(self):
        return len(self.image_list)

    def load_image(self, path):
        return None


def get_dataloader(opt):
    """获取数据加载器和预取迭代器。

    Args:
        opt: 配置选项

    Returns:
        tuple: 训练和验证数据的预取迭代器
    """
    # 创建数据集
    train_dataset = Dataset(
        phase="train",
        opt=opt,
        train_transform=train_transform,
        valid_transform=valid_transform,
    )
    valid_dataset = Dataset(
        phase="valid",
        opt=opt,
        train_transform=train_transform,
        valid_transform=valid_transform,
    )

    # 创建预取数据加载器
    train_dataloader = PrefetchDataLoader(
        num_prefetch_queue=opt.prefetch_queue_size,
        dataset=train_dataset,
        batch_size=opt.train_batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = PrefetchDataLoader(
        num_prefetch_queue=opt.prefetch_queue_size,
        dataset=valid_dataset,
        batch_size=opt.valid_batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    # 根据GPU可用性创建对应的预取器
    if len(opt.devices) > 0 and torch.cuda.is_available():
        train_prefetcher = CUDAPrefetcher(train_dataloader, opt)
        valid_prefetcher = CUDAPrefetcher(valid_dataloader, opt)
    else:
        train_prefetcher = CPUPrefetcher(train_dataloader)
        valid_prefetcher = CPUPrefetcher(valid_dataloader)

    # 创建可迭代的预取器
    train_iterator = PrefetcherIterator(
        train_prefetcher,
        length=len(train_dataset) // opt.train_batch_size,
    )
    valid_iterator = PrefetcherIterator(
        valid_prefetcher,
        length=len(valid_dataset) // opt.valid_batch_size,
    )

    return train_iterator, valid_iterator


if __name__ == "__main__":
    opt = get_option()

    # 获取预取迭代器
    train_iterator, valid_iterator = get_dataloader(opt)

    # 使用enumerate遍历预取的数据
    for i, batch in enumerate(train_iterator):
        print(f"Batch {i}:", batch["image"].shape, torch.unique(batch["label"]))
        if i >= 2:  # 仅示例几个批次
            break

    print("\nValidation data:")
    for i, batch in enumerate(valid_iterator):
        print(f"Batch {i}:", batch["image"].shape, torch.unique(batch["label"]))
        if i >= 2:  # 仅示例几个批次
            break
