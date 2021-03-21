
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def get_loader(dataset, pad_idx, shuffle=False, batch_size=1, ** params):
    collate_fn = PadCollate(pad_idx=pad_idx) if params.get("use_collate", True) else None
    loader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=params.get("n_workers", 8),
                        shuffle=shuffle,
                        pin_memory=params.get("pin_memory", True),
                        collate_fn=collate_fn)
    return loader


# padding each batch to be of  the same sequence length
class PadCollate:
    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs, captions = [], []
        references = []
        for (img, caption), reference in batch:
            imgs.append(img)
            captions.append(caption)
            references.append(reference)
        print(f'img shape = {imgs[0].shape}')
        imgs = torch.stack(imgs)  # dim=0
        print(f'imgs shape = {imgs.shape}')
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)  # (BXTX*), T-max_seq_len
        return (imgs, captions), references
