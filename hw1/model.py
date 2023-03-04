from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.nn import functional as F


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.rnn = nn.LSTM(
            embeddings.size(1),
            hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
            batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, num_class),
        )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        input = self.embed(batch['text'])
        packed_input = nn.utils.rnn.pack_padded_sequence(input, batch['length'], batch_first=True)
        self.rnn.flatten_parameters()
        packed_output, _ = self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = torch.mean(output, 1)
        output = self.fc(output)

        return output


class SeqTagging(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int
    ) -> None:
        super(SeqTagging, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.rnn = nn.LSTM(
            embeddings.size(1),
            hidden_size,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.cnn = nn.Sequential(nn.Conv1d(
            embeddings.size(1),
            embeddings.size(1),
            kernel_size=5,
            stride=1,
            padding=2),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, num_class)
        )


    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        input = self.embed(batch['tokens'])

        # input = input.permute(0, 2, 1)
        # input = self.cnn(input)
        # input = input.permute(0, 2, 1)

        packed_input = nn.utils.rnn.pack_padded_sequence(input, batch['length'], batch_first=True)
        self.rnn.flatten_parameters()
        packed_output, _ = self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output)

        return output
