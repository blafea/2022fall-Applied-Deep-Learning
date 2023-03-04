import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SeqTagDataset
from utils import Vocab
from model import SeqTagging


TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTagDataset] = {
        split: SeqTagDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_dataloader = DataLoader(dataset=datasets[TRAIN], batch_size=args.batch_size, shuffle=True, collate_fn=datasets[TRAIN].collate_fn)
    valid_dataloader = DataLoader(dataset=datasets[DEV], batch_size=args.batch_size, shuffle=True, collate_fn=datasets[DEV].collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    device = args.device
    print('DEVICE: ', device)
    model = SeqTagging(
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
        train_token_correct, train_token_total, valid_token_correct, valid_token_total = 0, 0, 0, 0
        train_joint_correct, train_joint_total, valid_joint_correct, valid_joint_total = 0, 0, 0, 0

        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        for batch in train_dataloader:
            batch['tokens'], batch['tags'], batch['mask'] = batch['tokens'].to(device), batch['tags'].to(device), batch['mask'].to(device)

            outputs = model(batch)
            batch_idx = torch.cat([torch.full((len,), i) for i, len in enumerate(batch['length'])])
            token_idx = torch.cat([torch.arange(0, len) for len in batch['length']])
            batch['mask'], batch['tags'] = batch['mask'][:, :outputs.size(1)], batch['tags'][:, :outputs.size(1)]
            loss = criterion(outputs[batch_idx, token_idx], batch['tags'][batch_idx, token_idx])
            outputs = outputs.argmax(-1, keepdim=True).squeeze(2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch['mask'], batch['tags'] = batch['mask'].cpu(), batch['tags'].cpu()
            batch['mask'] = batch['mask'][:, :batch['tags'].size(1)]
            correct = torch.eq(batch['tags'], outputs.cpu().view(batch['tags'].size()))
            correct = (correct * batch['mask']).sum(-1)

            train_token_correct += correct.sum().long().item()
            train_joint_correct += torch.eq(correct, batch['mask'].sum(-1)).sum().item()
            train_token_total += batch['mask'].sum().long().item()
            train_joint_total += len(batch['tags'])

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        with torch.no_grad():
            for batch in valid_dataloader:
                batch['tokens'], batch['tags'], batch['mask'] = batch['tokens'].to(device), batch['tags'].to(device), batch['mask'].to(device)
                outputs = model(batch)
                outputs = outputs.argmax(-1, keepdim=True).squeeze(2)
                batch['mask'], batch['tags'] = batch['mask'][:, :outputs.size(1)].cpu(), batch['tags'][:, :outputs.size(1)].cpu()
                batch['mask'] = batch['mask'][:, :batch['tags'].size(1)]
                correct = torch.eq(batch['tags'], outputs.cpu().view(batch['tags'].size()))
                correct = (correct * batch['mask']).sum(-1)

                valid_token_correct += correct.sum().long().item()
                valid_joint_correct += torch.eq(correct, batch['mask'].sum(-1)).sum().item()
                valid_token_total += batch['mask'].sum().long().item()
                valid_joint_total += len(batch['tags'])

            print('train joint: {:.6f} {}/{} token: {:.6f} {}/{} | valid joint: {:.6f} {}/{} token: {:.6f} {}/{}'.format(
                train_joint_correct/train_joint_total, 
                train_joint_correct, 
                train_joint_total,
                train_token_correct/train_token_total,
                train_token_correct,  
                train_token_total, 
                valid_joint_correct/valid_joint_total, 
                valid_joint_correct, 
                valid_joint_total, 
                valid_token_correct/valid_token_total, 
                valid_token_correct, 
                valid_token_total
            ))
            if valid_joint_correct/valid_joint_total > best_acc:
                best_acc = valid_joint_correct/valid_joint_total
                torch.save(model.state_dict(), args.ckpt_dir / "best_slot.pth")
                print('Saving model with acc {:.3f}'.format(best_acc))
        pass

    print("Best Accuracy:", best_acc)
    


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./"
    )

    # data
    parser.add_argument("--max_len", type=int, default=100)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
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
