import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_dataloader = DataLoader(dataset=datasets[TRAIN], batch_size=args.batch_size, shuffle=True, collate_fn=datasets[TRAIN].collate_fn)
    valid_dataloader = DataLoader(dataset=datasets[DEV], batch_size=args.batch_size, shuffle=True, collate_fn=datasets[DEV].collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    device = args.device
    print('DEVICE: ', device)
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        datasets[TRAIN].num_classes
    ).to(device)
    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        train_correct, train_total, train_loss, val_correct, val_total, val_loss = 0, 0, 0.0, 0, 0, 0.0

        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        for _, batch in enumerate(train_dataloader):
            batch['text'], batch['intent'] = batch['text'].to(device), batch['intent'].to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch['intent'])
            loss.backward()
            optimizer.step()
            train_pred = torch.argmax(outputs,  dim=1)
            train_total += batch['intent'].size(0)
            train_correct += (train_pred == batch['intent']).sum().item()
            train_loss += loss.item()

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        with torch.no_grad():
            for _, batch in enumerate(valid_dataloader):
                batch['text'], batch['intent'] = batch['text'].to(device), batch['intent'].to(device)
                outputs = model(batch)
                val_pred = torch.argmax(outputs, dim=1)
                val_total += batch['intent'].size(0)
                val_correct += (val_pred == batch['intent']).sum().item()
                loss = criterion(outputs, batch['intent'])
                val_loss += loss.item()

            print('Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} Loss: {:3.6f}'.format(
                train_correct/train_total,
                train_loss,
                val_correct/val_total,
                val_loss
            ))
            if val_correct/val_total > best_acc:
                best_acc = val_correct/val_total
                torch.save(model.state_dict(), args.ckpt_dir / "best_intent.pth")
                print('Saving model with acc {:.3f}'.format(val_correct/val_total))
        pass

    print("Best Accuracy:", best_acc)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./"
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=150)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
