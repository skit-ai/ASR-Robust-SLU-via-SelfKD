import torch
import pandas as pd
from functools import partial

class SnipsTtsDataset(torch.utils.data.Dataset):
    def __init__(self, split, use_n_best, sep_token):
        assert(split in ["train", "test"])
        super().__init__()
        self.df = pd.read_csv(f"snips_tts/snips_tts_asr_processed/{split}.csv")
        self.split = split
        self.use_n_best = use_n_best
        self.sep_token = sep_token

    def __getitem__(self, idx):
        clean = self.df["transcription"][idx]
        nbest = f" {self.sep_token} ".join(eval(self.df["hypothesis"][idx]))
        asr = eval(self.df["hypothesis"][idx])[0]
        return clean, asr, nbest, self.df["label"][idx]

    def __len__(self):
        return len(self.df)

def collate_fn(batch, tokenizer):
    inputs = {}

    clean = [item[0] for item in batch]
    clean_tk = tokenizer(clean, truncation=True, padding=True, return_tensors="pt")
    for k, v in clean_tk.items():
        inputs[f"clean_{k}"] = v

    asr = [item[1] for item in batch]
    nbest = [item[2] for item in batch]
    tag = [item[3] for item in batch]

    asr_tk = tokenizer(asr, truncation=True, padding=True, return_tensors="pt")
    nbest_tk = tokenizer(nbest, truncation=True, padding=True, return_tensors="pt")
    
    for k, v in asr_tk.items():
        inputs[f"asr_{k}"] = v        
    for k, v in nbest_tk.items():
        inputs[f"nbest_{k}"] = v        
    tag = torch.LongTensor(tag)
    return inputs, tag

def get_dataloaders(tokenizer, use_n_best, batch_size):
    tr_ds = SnipsTtsDataset("train", use_n_best, tokenizer.sep_token)
    ts_ds = SnipsTtsDataset("test", use_n_best, tokenizer.sep_token)

    tr_dl = torch.utils.data.DataLoader(
        tr_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )
    ts_dl = torch.utils.data.DataLoader(
        ts_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )
    return tr_dl, ts_dl