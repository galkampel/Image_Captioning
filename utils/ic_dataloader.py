
from torch.utils.data import DataLoader, random_split


def train_test_split(dataset, lengths):
    return random_split(dataset, lengths)


def get_loader(dataset, collate_fn, **dataloader_params):
    pad_idx = dataset.vocab.stoi['<PAD>']
    loader = DataLoader(dataset, batch_size=dataloader_params.get("BATCH_SIZE", 32),
                        num_workers=dataloader_params.get("N_WORKERS", 8),
                        shuffle=dataloader_params.get("shuffle", True),
                        pin_memory=dataloader_params.get("PIN_MEMEORY", True),
                        collate_fn=collate_fn(pad_idx=pad_idx),)
    return loader


