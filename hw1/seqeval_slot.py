import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch, csv
from torch.utils.data import DataLoader
from dataset import SeqTagDataset
from model import SeqTagging
from utils import Vocab

from seqeval.scheme import IOB2
from seqeval.metrics import classification_report

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTagDataset(data, vocab, tag2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SeqTagging(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes
    ).to(device)

    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)

    # TODO: predict dataset
    with torch.no_grad():
        test_ids = []
        test_tags = []
        test_lens = []
        true_tags = []
        for i, batch in enumerate(test_dataloader):
            batch['tokens'], batch['tags'], batch['mask'] = batch['tokens'].to(device), batch['tags'].to(device), batch['mask'].to(device)
            test_ids += batch['id']
            outputs = model(batch)
            outputs = outputs.argmax(-1, keepdim=True).squeeze(2)
            batch['mask'], batch['tags'] = batch['mask'][:, :outputs.size(1)], batch['tags'][:, :outputs.size(1)]
            test_tags += outputs.cpu().tolist()
            test_lens += batch['mask'].sum(-1).long().cpu().tolist()
            true_tags += batch['ori']
        out_tags = [[]]
        i = 0
        for tags, lens in zip(test_tags, test_lens):
            for id, tag in enumerate(tags):
                if id < lens - 1:
                    out_tags[i].append(dataset.idx2label(tag))
                else:
                    out_tags[i].append(dataset.idx2label(tag))
                    i += 1
                    out_tags.append([])
                    break

        print(classification_report(true_tags, out_tags[:-1], mode='strict', scheme=IOB2))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/eval.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/slot/tag.pth",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
