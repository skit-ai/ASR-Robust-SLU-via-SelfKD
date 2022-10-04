import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchmetrics import F1Score
import json
import os
os.makedirs("./stats/", exist_ok=True)
os.makedirs("./stats/dstc2", exist_ok=True)
os.system("rm -rf lightning_logs/")

class Experiment(pl.LightningModule):
    def __init__(self, use_clean, use_n_best):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=21)
        self.use_clean = use_clean
        self.use_n_best = use_n_best
        
        if self.use_clean and self.use_n_best:
            raise ValueError("Invalid args.")

        self.train_f1 = F1Score(21, average="macro")
        self.val_clean_f1 = F1Score(21, average="macro")
        self.val_asr_f1 = F1Score(21, average="macro")

        self.best = {
            "clean_f1": 0.,
            "asr_f1": 0.
        }

    def training_step(self, batch, batch_idx):
        all_inputs, y = batch
        if self.use_clean:
            x = all_inputs["clean_input_ids"]
            mask = all_inputs["clean_attention_mask"]
        elif self.use_n_best:
            x = all_inputs["nbest_input_ids"]
            mask = all_inputs["nbest_attention_mask"]
        else:
            x = all_inputs["asr_input_ids"]
            mask = all_inputs["asr_attention_mask"]
        
        logits = self.model(input_ids=x, attention_mask=mask).logits
        loss = F.binary_cross_entropy_with_logits(logits, y)

        self.train_f1(logits.sigmoid(), y.long())

        self.log("train/LOSS", loss.item())
        self.log("train/F1", self.train_f1)
        return loss
    
    def validation_step(self, batch, batch_idx):
        all_inputs, y = batch
        if self.use_n_best:
            x = all_inputs["nbest_input_ids"]
            mask = all_inputs["nbest_attention_mask"]
        else:
            x = all_inputs["asr_input_ids"]
            mask = all_inputs["asr_attention_mask"]
        
        logits = self.model(input_ids=x, attention_mask=mask).logits
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("val/LOSS", loss.item())
        self.val_asr_f1(logits.sigmoid(), y.long())
    
        x = all_inputs["clean_input_ids"]
        mask = all_inputs["clean_attention_mask"]
        
        logits = self.model(input_ids=x, attention_mask=mask).logits
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("val/Clean_LOSS", loss.item())
        self.val_clean_f1(logits.sigmoid(), y.long())

    # def on_train_epoch_end(self):
    #     self.train_f1.reset()

    def on_validation_epoch_end(self):
        self.log("val/F1", self.val_asr_f1)
        self.log("val/Clean_F1", self.val_clean_f1)
        asr_f1 = self.val_asr_f1.compute()
        clean_f1 = self.val_clean_f1.compute()
        if asr_f1 > self.best["asr_f1"]:
            self.best["asr_f1"] = asr_f1.item()
            self.best["clean_f1"] = clean_f1.item()
        # self.val_asr_f1.reset()
        # self.val_clean_f1.reset()
        
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        schd = torch.optim.lr_scheduler.MultiStepLR(opt, [25, 50])
        return [opt], [schd]

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pytorch_lightning.callbacks import EarlyStopping
    from data import get_dataloaders

    parser = ArgumentParser()
    parser.add_argument("--use_n_best", action="store_true", default=False)
    parser.add_argument("--use_clean", action="store_true", default=False)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    tr_dl, ts_dl = get_dataloaders(tokenizer, args.use_n_best, 32)
    exp = Experiment(args.use_clean, args.use_n_best)

    checkpointer = EarlyStopping(monitor="val/F1", mode="max", patience=10)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100,
        callbacks=[checkpointer],
        precision=16,
        # gradient_clip_val=50,
        accumulate_grad_batches=1
    )
    trainer.fit(exp, train_dataloaders=tr_dl, val_dataloaders=ts_dl)
    from uuid import uuid4
    to_save = exp.best
    to_save["use_clean"] = args.use_clean
    to_save["use_n_best"] = args.use_n_best
    json.dump(to_save, open(f"stats/dstc2/{str(uuid4())}.json", "w"))