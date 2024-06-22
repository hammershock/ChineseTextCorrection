import difflib
from typing import List, Tuple, Sequence, Iterator, TypeVar

import numpy as np
import torch
from joblib import Memory
from pypinyin import lazy_pinyin
from torch.utils.data import TensorDataset, Dataset
from tqdm import tqdm
from transformers import BertTokenizer

from utils import batchify, load_json, pad_to

memory = Memory(location="./cache", verbose=0)


def load_data_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [line.strip() for line in f.readlines()]
    return data


_T = TypeVar("_T")


def _generate_data(correct_lines: List[str], tokenizer: BertTokenizer):
    for corrected in tqdm(correct_lines, "building dataset"):
        correct_tokens = tokenizer.tokenize(corrected)
        # add special token
        # special_token_ids = [1] + [0] * len(correct_tokens) + [1]
        correct_tokens = ["[CLS]"] + list(correct_tokens) + ["[SEP]"]
        correct_tokens_ids = tokenizer.convert_tokens_to_ids(correct_tokens)
        yield correct_tokens_ids


def _generate_data_collate(correct_lines: List[str],
                           tokenizer: BertTokenizer, max_len=128, overlap=64) -> Iterator[
    Tuple[np.ndarray, ...]]:
    for correct_token_ids in _generate_data(correct_lines, tokenizer):
        correct_token_ids, attn_mask = pad_to(correct_token_ids, max_len, tokenizer.pad_token_id)
        # batches = batchify(correct_token_ids, special_token_ids, max_len=max_len, overlap=overlap, pad=['[PAD]', 1])
        # for (batch_correct_token_ids, batched_special_token_ids), attn_mask in batches:
        #     yield batch_correct_token_ids, attn_mask, batched_special_token_ids
        yield correct_token_ids, attn_mask


def _make_dataset(correct_lines, tokenizer, max_len=128, overlap=64):
    token_ids, attn_mask = zip(*_generate_data_collate(correct_lines, tokenizer, max_len=max_len, overlap=overlap))
    token_ids = torch.LongTensor(token_ids)
    attn_mask = torch.LongTensor(attn_mask)
    return token_ids, attn_mask


@memory.cache
def make_dataset(data_path, tokenizer_path, pinyin_vocab_path, **kwargs):
    """
    生成数据集，使用joblib缓存机制。注意当数据和词表内部发生变更时要清理缓存
    :param data_path: tsv数据路径，第一列原始文本，第二列改正后文本
    :param tokenizer_path:
    :param pinyin_vocab_path: json文件
    :param kwargs: max_len, overlap等参数
    :return:
    """
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    with open(data_path, "r", encoding="utf-8") as f:
        parts = [line.strip().split("\t") for line in f.readlines()]
        _, corrected = zip(*parts)
        token_ids, attn_mask = _make_dataset(corrected, tokenizer, **kwargs)
    return token_ids, attn_mask


class MLMDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, pinyin_vocab_path, **kwargs):
        token_ids, attn_mask = make_dataset(data_path, tokenizer_path, pinyin_vocab_path, **kwargs)
        self.token_ids = token_ids
        self.attn_mask = attn_mask

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.token_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attn_mask[idx], dtype=torch.long),
        }
