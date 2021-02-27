
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def get_loader(dataset, use_collate=True, **dataloader_params):
    pad_idx = dataset.vocab.stoi['<PAD>']
    collate_fn = PadCollate(pad_idx=pad_idx) if use_collate else None
    loader = DataLoader(dataset, batch_size=dataloader_params.get("BATCH_SIZE", 32),
                        num_workers=dataloader_params.get("N_WORKERS", 8),
                        shuffle=dataloader_params.get("shuffle", True),
                        pin_memory=dataloader_params.get("PIN_MEMEORY", True),
                        collate_fn=collate_fn)
    return loader


# padding each batch to be of  the same sequence length
class PadCollate:
    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs, captions = [] , []
        for img, caption in batch:
            imgs.append(img)
            captions.append(caption)
        imgs = torch.stack(imgs)  # dim=0
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)  # (BXTX*), T-max_seq_len
        return imgs, captions
