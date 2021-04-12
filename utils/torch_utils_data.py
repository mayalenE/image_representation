import torch
from torch.utils.data import DataLoader, Subset
from exputils.seeding import set_seed


def get_dataloader(dataset, dataloader_config):
    if (dataloader_config.num_workers > 0):
        # if dataloader's num_workers > 0:
        #    - use multiprocessing for reader hdf5 file in parallel 
        #    - set the same seed in every process for reproducibility
        torch.multiprocessing.set_start_method("fork", force=True)

    if dataloader_config.ids is not None:
        subset_ids = dataloader_config.ids
    else:
        subset_ids = range(len(dataset))
    # subset_sampler = torch.utils.data.sampler.SubsetRandomSampler(subset_ids)
    subset_dataset = Subset(dataset, subset_ids)
    dataloader = DataLoader(subset_dataset,
                            # sampler = subset_sampler,
                            batch_size=dataloader_config.batch_size,
                            shuffle=dataloader_config.shuffle,
                            num_workers=dataloader_config.num_workers,
                            worker_init_fn=set_seed
                            )

    return dataloader
