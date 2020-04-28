import itertools


class ConcatLoader:
    def __init__(self, dataloaders):
        self.loaders = dataloaders

    def __iter__(self):
        self.iters = [iter(loader) for loader in self.loaders]
        self.idx_cycle = itertools.cycle(list(range(len(self.loaders))))
        return self

    def __next__(self):
        loader_idx = next(self.idx_cycle)
        loader = self.iters[loader_idx]
        batch = next(loader)
        dataset = loader._dataset
        dat_name = dataset.pose_dataset.name
        ret_batch = {}
        ret_batch["dataset"] = dat_name
        ret_batch["split"] = dataset.pose_dataset.split
        if dataset.sample_nb == 1:
            ret_batch["supervision"] = ["data"]
        else:
            ret_batch["supervision"] = ["consist"]
        ret_batch["data"] = batch
        return ret_batch

    def __len__(self):
        return min(len(loader) for loader in self.loaders) * len(self.loaders)
