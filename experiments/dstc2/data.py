import torch
from functools import partial
import json
import os
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append(".")

root = "data/"
class Dstc2Dataset(torch.utils.data.Dataset):
    def __init__(self, split, use_n_best, sep_token, label_encoder=None):
        assert(split in ["train", "test"])
        super().__init__()
        if split != "train":
            assert(label_encoder is not None)
            self.label_encoder = label_encoder
        else:
            self.label_encoder = LabelEncoder()
        self.split = split
        self.use_n_best = use_n_best
        self.sep_token = sep_token

        flist = open(f"scripts/config/dstc2_{split}.flist").read().split("\n")
        flist.remove("")
        
        self.all_clean = []
        self.all_nbest = []
        self.all_labels = []
        labels_flattened = []
        for p in flist:
            c, nb, l = self.read_one(p, label_encoder)
            self.all_clean.extend(c)
            self.all_nbest.extend(nb)
            self.all_labels.extend(l)
            if self.split == "train":
                for item in l:
                    labels_flattened.extend(item)

        if self.split == "train":
            self.label_encoder = self.label_encoder.fit(labels_flattened)
        self.all_labels = [self.label_encoder.transform(item) for item in self.all_labels]
        self.num_labels = len(self.label_encoder.classes_)
        print("Num Labels", self.num_labels)

    def one_hot(self, size, hots):
        x = torch.zeros((size,))
        x[hots] = 1.0
        return x.float()

    def read_one(self, path, label_encoder=None):
        log = json.load(open(os.path.join(root, path, "log.json"), "r"))
        label = json.load(open(os.path.join(root, path, "label.json"), "r"))

        utterances_gt = []
        nbest = []
        slots = []

        for it, lt in zip(log["turns"], label["turns"]):
            sl = []
            for s in lt["semantics"]["json"]:
                if len(s['slots']) == 0:
                    to_add_slot = f"{s['act']}"
                else:
                    to_add_slot = f"{s['act']}-{s['slots'][0][0]}"
                if label_encoder is None or (label_encoder is not None and to_add_slot in label_encoder.classes_):
                    sl.append(to_add_slot)
            if len(sl) == 0:
                continue
            utterances_gt.append(lt["transcription"])
            nbest.append([item["asr-hyp"] for item in it["input"]["live"]["asr-hyps"][:5]])
            slots.append(sl)
        return utterances_gt, nbest, slots

    def __getitem__(self, idx):
        clean = self.all_clean[idx]
        nbest = f"{self.sep_token}".join(self.all_nbest[idx])
        asr = self.all_nbest[idx][0]
        return clean, asr, nbest, self.one_hot(self.num_labels, self.all_labels[idx])

    def __len__(self):
        return len(self.all_clean)

def collate_fn(batch, tokenizer):
    inputs = {}

    clean = [item[0] for item in batch]
    clean_tk = tokenizer(clean, truncation=True, padding=True, return_tensors="pt")
    for k, v in clean_tk.items():
        inputs[f"clean_{k}"] = v

    asr = [item[1] for item in batch]
    nbest = [item[2] for item in batch]
    tag = [item[3].unsqueeze(0) for item in batch]

    asr_tk = tokenizer(asr, truncation=True, padding=True, return_tensors="pt")
    nbest_tk = tokenizer(nbest, truncation=True, padding=True, return_tensors="pt")
    
    for k, v in asr_tk.items():
        inputs[f"asr_{k}"] = v        
    for k, v in nbest_tk.items():
        inputs[f"nbest_{k}"] = v
    tag = torch.cat(tag, dim=0)
    return inputs, tag

def get_dataloaders(tokenizer, use_n_best, batch_size):
    tr_ds = Dstc2Dataset("train", use_n_best, tokenizer.sep_token)
    ts_ds = Dstc2Dataset("test", use_n_best, tokenizer.sep_token, tr_ds.label_encoder)

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