"""
build pinyin vocab
获取拼音词汇表，生成文件`../pinyin_vocabulary.json`
"""
import json

from pypinyin import lazy_pinyin


def generate_all_pinyin():
    # 获取所有的汉字
    all_chinese_chars = ''.join(chr(i) for i in range(0x4e00, 0x9fff + 1))
    pinyin_set = set()

    # 遍历所有汉字，获取拼音
    for char in all_chinese_chars:
        for py in lazy_pinyin(char):
            if py != char:
                pinyin_set.add(py)

    return pinyin_set


pinyins = generate_all_pinyin()

pinyins = sorted(pinyins)
pinyin_vocab = {"[OTHER]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3}
for pinyin in pinyins:
    pinyin_vocab[pinyin] = len(pinyin_vocab)

print(len(pinyin_vocab))

with open('../config/pinyin_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(pinyin_vocab, f, ensure_ascii=False, indent=4)
