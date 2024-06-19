import difflib
import json
from typing import List, Tuple, Sequence, Iterator, TypeVar

import numpy as np
import torch
from pypinyin import lazy_pinyin
from torch.utils.data import TensorDataset
from joblib import Memory
from tqdm import tqdm
from transformers import BertTokenizer

from utils import batchify


memory = Memory(location="./cache", verbose=0)


def load_data_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [line.strip() for line in f.readlines()]
    return data


_T = TypeVar("_T")


def generate_difference_mask(original: Sequence[_T], corrected: Sequence[_T], null_fill: _T) -> Tuple[List[int], List[_T]]:
    """比较两个序列的不同, 生成一个与original序列相同形状的mask序列, 0的位置代表正确, 1的位置代表错误"""
    mask = [0] * len(original)
    corrected_original = [null_fill] * len(original)
    matcher = difflib.SequenceMatcher(None, original, corrected)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        for i, j in zip(range(i1, i2), range(j1, j2)):
            if tag != 'equal':
                mask[i] = 1
            corrected_original[i] = corrected[j]
    assert len(original) == len(mask)
    return mask, corrected_original


def pinyin_tokenize(text: str, tokenizer: BertTokenizer, pad_value="[OTHER]") -> Tuple[np.array, np.array]:
    """输入文本，生成相等长度的文本tokens序列和pinyin tokens序列"""
    tokens = np.array(tokenizer.tokenize(text), dtype="<U10")
    pinyins = np.full_like(tokens, dtype="<U10", fill_value=pad_value)
    mask_ishanzi = (tokens >= '\u4e00') & (tokens <= '\u9fff') | (tokens == '\u3007')
    pinyins[mask_ishanzi] = lazy_pinyin(tokens[mask_ishanzi].tolist())
    assert len(tokens) == len(pinyins)
    return tokens, pinyins


def _generate_data(text_lines: List[str], correct_lines: List[str], tokenizer: BertTokenizer):
    for original, corrected in tqdm(zip(text_lines, correct_lines), "building dataset", total=len(text_lines)):
        tokens, pinyins = pinyin_tokenize(original, tokenizer, pad_value="[OTHER]")
        correct_tokens, _ = pinyin_tokenize(corrected, tokenizer, pad_value="[OTHER]")

        # add special token
        tokens = ['[CLS]'] + list(tokens) + ['[SEP]']
        pinyins = ["[CLS]"] + list(pinyins) + ["[SEP]"]
        correct_tokens = ["[CLS]"] + list(correct_tokens) + ["[SEP]"]

        correction_mask, corrected_tokens = generate_difference_mask(tokens, correct_tokens, '[BLANK]')
        yield tokens, corrected_tokens, pinyins, correction_mask


def _generate_data_collate(text_lines: List[str], correct_lines: List[str],
                           tokenizer: BertTokenizer, pinyin_vocab, max_len=128, overlap=64) -> Iterator[Tuple[np.ndarray, ...]]:
    for seqs in _generate_data(text_lines, correct_lines, tokenizer):
        batches = batchify(*seqs, max_len=max_len, overlap=overlap, pad=['[PAD]', '[PAD]', '[PAD]', 0])
        for (tokens, corrected_tokens, py_tokens, correct_mask), attn_mask in batches:
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            corrected_token_ids = tokenizer.convert_tokens_to_ids(corrected_tokens)
            py_token_ids = [pinyin_vocab[item] for item in py_tokens]
            yield token_ids, corrected_token_ids, py_token_ids, correct_mask, attn_mask


def _make_dataset(original_lines, correct_lines, tokenizer, pinyin_vocab, max_len=128, overlap=64):
    token_ids, corrected_ids, py_token_ids, correct_mask, attn_mask = zip(*_generate_data_collate(original_lines, correct_lines, tokenizer, pinyin_vocab, max_len=max_len, overlap=overlap))
    token_ids = torch.LongTensor(token_ids)
    corrected_ids = torch.LongTensor(corrected_ids)
    py_token_ids = torch.LongTensor(py_token_ids)
    correct_mask = torch.FloatTensor(correct_mask)
    attn_mask = torch.LongTensor(attn_mask)
    return TensorDataset(token_ids, corrected_ids, py_token_ids, correct_mask, attn_mask)


@memory.cache
def make_dataset(data_path, tokenizer_path, pinyin_vocab_path, **kwargs):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    with open(pinyin_vocab_path, "r", encoding="utf-8") as f:
        pinyin_vocab = json.load(f)
    with open(data_path, "r", encoding="utf-8") as f:
        parts = [line.strip().split("\t") for line in f.readlines()]
        original, corrected = zip(*parts)
    return _make_dataset(original, corrected, tokenizer, pinyin_vocab, **kwargs)


if __name__ == '__main__':
    original_texts = ["你好玛，你的名子角什么牙,,"]
    correct_texts = ["你好啊, 你的名字叫什么呢"]
    result = generate_difference_mask(original_texts[0], correct_texts[0], '[BLANK]')
    print(result)
