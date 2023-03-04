from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab, pad_to_len
import torch

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        samples.sort(key=lambda x: len(x['text'].split()), reverse=True)
        batch = dict()
        batch['id'] = [sample['id'] for sample in samples]
        batch['text'] = [sample['text'].split() for sample in samples]
        batch['length'] = [len(s) for s in batch['text']]
        batch['text'] = torch.tensor(self.vocab.encode_batch(batch['text'], to_len=self.max_len))
        try:
            batch['intent'] = torch.tensor([self.label2idx(sample['intent']) for sample in samples], dtype=torch.long)
        except KeyError:
            batch['intent'] = []

        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTagDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: tag for tag, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        samples.sort(key=lambda x: len(x['tokens']), reverse=True)
        batch = dict()
        batch['id'] = [sample['id'] for sample in samples]
        batch['tokens'] = [sample['tokens'] for sample in samples]
        batch['length'] = [len(s) for s in batch['tokens']]
        batch['tokens'] = torch.tensor(self.vocab.encode_batch(batch['tokens'], to_len=self.max_len))
        try:
            batch['tags'] = [[self.label2idx(tag) for tag in sample['tags']] for sample in samples]
            batch['tags'] = torch.tensor(pad_to_len(batch['tags'], self.max_len, 0), dtype=torch.long)
            batch['ori'] = [sample['tags'] for sample in samples]
        except KeyError:
            batch['tags'] = torch.tensor([[0] * self.max_len] * len(samples))

        batch['mask'] = batch['tokens'].gt(0).clone().detach()
        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
