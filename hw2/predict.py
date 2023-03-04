import argparse
import json
from itertools import chain
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    BertTokenizerFast,
    BertForQuestionAnswering,
    SchedulerType,
    default_data_collator,
)

accelerator = Accelerator()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    parser.add_argument(
        "--test_file", type=str, default="test.json", help="A csv or a json file containing the testing data."
    )
    parser.add_argument(
        "--context_file", type=str, default="context.json", help="A csv or a json file containing the training data."
    )
    parser.add_argument("--result_file", default="result.csv")
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=1, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        default=True,
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--cs_model",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="./cs_model",
    )
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument(
        "--qa_model",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="./qa_model",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    

    args = parser.parse_args()

    return args

def data_collator(features):
    """
    :param features: list of data instance
    :return: torch.Tensor in batch
    """
    first = features[0]
    batch = {}

    # Special handling for ids and paragraphs
    ids = [feature.pop("id") for feature in features]
    paragraphs = [feature.pop("paragraphs") for feature in features]

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
    dtype = torch.long if isinstance(label, int) else torch.float
    batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    batch["ids"] = ids
    batch["paragraphs"] = paragraphs

    return batch

def main():
    args = parse_args()

    # Context Selection

    # Model and Tokenizer
    accelerator_log_kwargs = {}

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    config = AutoConfig.from_pretrained(args.cs_model, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.cs_model, use_fast=True, cache_dir=args.cache_dir)
    cs_model = AutoModelForMultipleChoice.from_pretrained(
        args.cs_model,
        config=config,
        cache_dir=args.cache_dir,
    )
    cs_model.to(DEVICE)
    cs_model.resize_token_embeddings(len(tokenizer))
    with open(args.context_file, 'r', encoding="utf-8") as f:
        context_json = json.load(f)
    context_name = "question"
    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples[context_name]]
        second_sentences = [
            [context_json[i] for i in idx] for idx in examples["paragraphs"]
        ]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=args.max_length,
            padding="max_length",
            truncation=True,
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        tokenized_inputs['label'] = [0 for _ in examples["paragraphs"]]
        return tokenized_inputs


    # Dataset
    data_files = {}
    data_files["test"] = args.test_file
    extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=args.cache_dir)
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function, batched=True, load_from_cache_file=True
        )
    test_dataset = processed_datasets["test"]
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_batch_size)

    # Testing
    all_ids = []
    all_paragraphs = []
    all_pred = []

    for step, batch in enumerate(tqdm(test_dataloader)):
        ids = batch.pop("ids")
        paragraphs = batch.pop("paragraphs")
        _ = batch.pop("labels")
        data = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = cs_model(**data)
        predicted = outputs.logits.argmax(dim=-1)

        all_ids += ids
        all_paragraphs += paragraphs
        all_pred.append(predicted)

    all_pred = torch.cat(all_pred).cpu().numpy()
    selected_context = {instance_id: paragraphs[context_idx]
                        for instance_id, context_idx, paragraphs in zip(all_ids, all_pred, all_paragraphs)}

    # Question Answering

    # Model and Tokenizer
    qa_model = BertForQuestionAnswering.from_pretrained(args.qa_model, cache_dir=args.cache_dir).to(DEVICE)
    tokenizer = BertTokenizerFast.from_pretrained(args.qa_model, cache_dir=args.cache_dir)

    with open(args.test_file, 'r') as f:
        test_questions = json.load(f)

    test_questions_tokenized = tokenizer([test_question["question"] for test_question in test_questions], add_special_tokens=False)
    context_tokenized = tokenizer(context_json, add_special_tokens=False)

    ## Dataset and Dataloader
    class QA_Dataset(Dataset):
        def __init__(self, split, cs_pred, tokenized_questions, tokenized_paragraphs):
            self.split = split
            self.cs_pred = cs_pred
            self.tokenized_questions = tokenized_questions
            self.tokenized_paragraphs = tokenized_paragraphs
            self.max_question_len = 250
            self.max_paragraph_len = 250
            self.doc_stride = 128
            self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

        def __len__(self):
            return len(self.cs_pred)

        def __getitem__(self, idx):
            question = self.cs_pred[idx]
            tokenized_question = self.tokenized_questions[idx]
            tokenized_paragraph = self.tokenized_paragraphs[question]

            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []

            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i: i + self.max_paragraph_len] + [102]

                # Pad sequence and obtain inputs to model
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)

                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)

            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

        def padding(self, input_ids_question, input_ids_paragraph):
            # Pad zeros if sequence length is shorter than max_seq_len
            padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
            # Indices of input sequence tokens in the vocabulary
            input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
            # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
            token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
            # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
            attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len

            return input_ids, token_type_ids, attention_mask

    test_set = QA_Dataset("test", list(selected_context.values()), test_questions_tokenized, context_tokenized)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

    def match_quote(answer, left_symbol, right_symbol):
        answer = answer.replace(' ', '')
        left_quote = answer.count(left_symbol)
        right_quote = answer.count(right_symbol)
        if left_quote - right_quote == 1:
            return answer + right_symbol
        elif left_quote - right_quote == -1:
            return left_symbol + answer
        else:
            return answer

    def evaluate(data, output, relevant):
        answer = ''
        max_prob = float('-inf')
        num_of_windows = data[0].shape[1]

        for k in range(num_of_windows):
            # Obtain answer by choosing the most probable start position / end position
            start_prob, start_index = torch.max(output.start_logits[k], dim=0)
            end_prob, end_index = torch.max(output.end_logits[k], dim=0)

            prob = start_prob + end_prob

            if prob >= max_prob:
                max_prob = prob
                answer = tokenizer.decode(data[0][0][k][start_index: end_index + 1])
                previous = tokenizer.decode(data[0][0][k][0: start_index])

        previous = previous.replace(' ', '')
        answer = answer.replace(' ', '')

        answer = match_quote(answer, '「', '」')
        answer = match_quote(answer, '《', '》')
        answer = match_quote(answer, '〈', '〉')

        return answer

    result = []

    qa_model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            output = qa_model(input_ids=data[0].squeeze(dim=0).to(DEVICE), token_type_ids=data[1].squeeze(dim=0).to(DEVICE),
                           attention_mask=data[2].squeeze(dim=0).to(DEVICE))

            result.append(evaluate(data, output, list(selected_context.values())[i]))

    with open(args.result_file, 'w') as f:
        f.write("ID,Answer\n")
        for i, test_question in enumerate(test_questions):
            f.write(f"{test_question['id']},{result[i].replace(',','')}\n")



if __name__ == "__main__":
    main()