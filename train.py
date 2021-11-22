import os
from datetime import datetime
from itertools import takewhile
from tqdm import tqdm, trange

import torch
from torch import optim
from torch.nn import DataParallel
# from torch.nn import CrossEntropyLoss  # PyTorch < 1.10.0 doesn't support label smoothing
from torch.utils.data import DataLoader

from lib.nn import CrossEntropyLoss
from lib.optim.lr_scheduler import NoamLR

from dataset import Vocab, DigitDataset, TypoDataset
from model import TransformerModel


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device="cuda", parallel=False):
        self.model = DataParallel(model) if parallel else model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parallel = parallel
        self.device = torch.device(device)
        self.model.to(device)
        if parallel:
            self.model_state_dict = self.model.module.state_dict
        else:
            self.model_state_dict = self.model.state_dict

    def run_epoch(self, dataloader, train=True):
        total_loss, total_corrects, total_samples = 0, 0, 0
        if train:
            desc = "Train"
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            desc = "Test"
            self.model.eval()
            torch.set_grad_enabled(False)

        for src, tgt, src_mask, tgt_mask in tqdm(dataloader, desc=desc, ncols=85, leave=False):
            src, tgt = src.to(self.device), tgt.to(self.device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            src_mask = None if src_mask is None else src_mask.to(self.device)
            tgt_mask = None if tgt_mask is None else tgt_mask[:, 1:].to(self.device)

            out = self.model(src, tgt_in, src_mask, tgt_mask)
            loss = self.criterion(out.permute(0, 2, 1), tgt_out)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            pred = torch.argmax(out, dim=2)
            corrects = torch.sum(torch.all(pred == tgt_out, dim=1)).item()

            total_loss += loss.detach().item() * src.size(0)
            total_corrects += corrects
            total_samples += src.size(0)

        epoch_loss = total_loss / total_samples
        epoch_acc = total_corrects / total_samples
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def predict(self, src, src_mask, start_idx, end_idx):
        self.model.eval()
        pred = torch.full((src.size(0), 1), start_idx, dtype=torch.long, device=self.device)
        for _ in range(src.size(1) + 2):
            out_t = self.model(src, pred, src_mask, None)
            pred_t = torch.argmax(out_t[:, -1, :], dim=1)
            pred = torch.hstack((pred, pred_t.unsqueeze(1)))

        end_lambda = lambda x: x != end_idx
        results = []
        for i in range(src.size(0)):
            src_str = "".join(map(Vocab.to_vocab, takewhile(end_lambda, src[i, 1:].tolist())))
            pred_str = "".join(map(Vocab.to_vocab, takewhile(end_lambda, pred[i, 1:].tolist())))
            results.append((src_str, pred_str))
        return results

    def train(self, train_loader, valid_loader, num_epochs, ex_src=None, ex_src_mask=None, ex_len=85):
        ex_src = None if ex_src is None else ex_src.to(self.device)
        ex_src_mask = None if ex_src_mask is None else ex_src_mask.to(self.device)
        start_idx = Vocab.to_idx(train_loader.dataset.start_token)
        end_idx   = Vocab.to_idx(train_loader.dataset.end_token)

        best_acc, best_state_dict = 0, None
        for e in trange(num_epochs, desc="Epoch", ncols=85):
            train_loss, train_acc = self.run_epoch(train_loader, train=True)
            tqdm.write(f"Epoch {e+1:3d} Train Loss: {train_loss:.5f} Acc: {train_acc * 100:6.2f}")
            test_loss, test_acc = self.run_epoch(valid_loader, train=False)
            tqdm.write(f"Epoch {e+1:3d} Test  Loss: {test_loss:.5f} Acc: {test_acc * 100:6.2f}")

            if ex_src is not None:
                results = self.predict(ex_src, ex_src_mask, start_idx, end_idx)
                for i, (src_str, pred_str) in enumerate(results):
                    sep = "\n      " if len(src_str) + len(pred_str) > ex_len - 14 else ""
                    tqdm.write(f"     Ex {i} {src_str}{sep} => {pred_str}")
            if self.scheduler is not None:
                self.scheduler.step()
            if test_acc > best_acc:
                best_acc, best_state_dict = test_acc, self.model_state_dict()

        return best_acc, best_state_dict


def main(args):
    if len(args.gpus) == 1 and args.gpus[0] == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "digit" or args.dataset == "d":
        args.dataset == "digit"
        train_dataset = DigitDataset(num_data=10000, token_len=32, seed=109)
        test_dataset  = DigitDataset(num_data=1000,  token_len=32, seed=42)
        collate_fn = DigitDataset.collate_fn
        vocab_size = DigitDataset.vocab_size

        train_loader = DataLoader(train_dataset, batch_size=100, collate_fn=collate_fn, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=100, collate_fn=collate_fn, shuffle=False)

        model = TransformerModel(vocab_size, d_model=64, dim_feedforward=256,
                                 num_encoder_layers=2, num_decoder_layers=2)
        criterion = CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=5e-3)
        scheduler = None

        ex_batch = next(iter(test_loader))
        ex_src, ex_src_mask = ex_batch[0][:5].to(device), None
        parallel = False
        num_epochs = 100

    elif args.dataset.startswith("typo") or args.dataset.startswith("t"):
        if args.dataset == "typo-word" or args.dataset == "tw":
            level = "word"
            args.dataset == "typo-word"
        elif args.dataset == "typo-sentence" or args.dataset == "ts":
            level = "sentence"
            args.dataset == "typo-sentence"

        train_dataset = TypoDataset(num_data=51200, level=level, split="train", seed=109)
        test_dataset  = TypoDataset(num_data=10240, level=level, split="valid", seed=42)
        collate_fn = TypoDataset.collate_fn
        vocab_size = TypoDataset.vocab_size

        train_loader = DataLoader(train_dataset, batch_size=256, collate_fn=collate_fn,
                                  num_workers=4, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=256, collate_fn=collate_fn,
                                  num_workers=4, shuffle=False)

        if args.dataset == "typo-word":
            model = TransformerModel(vocab_size, d_model=512, dim_feedforward=2048,
                                     num_encoder_layers=4, num_decoder_layers=4)
            num_epochs = 200
        elif args.dataset == "typo-sentence":
            model = TransformerModel(vocab_size, d_model=256, dim_feedforward=1024,
                                     num_encoder_layers=4, num_decoder_layers=4)
            num_epochs = 300

        criterion = CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        scheduler = NoamLR(optimizer, warmup_epochs=30)

        ex_batch = next(iter(test_loader))
        ex_src, ex_src_mask = ex_batch[0][:5].to(device), ex_batch[2][:5].to(device)
        parallel = True

    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    parallel = parallel and len(args.gpus) > 1
    trainer = Trainer(model, criterion, optimizer, scheduler, device, parallel)
    base_acc, state_dict = trainer.train(train_loader, test_loader, num_epochs,
                                         ex_src=ex_src, ex_src_mask=ex_src_mask)

    os.makedirs(f"./ckpt/{args.dataset}", exist_ok=True)
    filename = f"./ckpt/{args.dataset}/{datetime.now().strftime('%Y%m%d%H%M%S')}.pt"
    torch.save(state_dict, filename)
    print(f"Best Acc: {base_acc * 100:6.2f}%")
    print(f"Checkpoint saved to '{filename}'")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-g", "--gpus",    type=int, nargs="*")
    parser.add_argument("-d", "--dataset", required=True,
                        choices=["d", "digit", "tw", "typo-word", "ts", "typo-sentence"])
    args = parser.parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))

    try:
        main(args)
    except KeyboardInterrupt:
        pass
