from collections import defaultdict, OrderedDict
from typing import Sequence, Iterator, Tuple, Dict, TypeVar

import numpy as np
from fuzzywuzzy import process
from tqdm import tqdm
from pypinyin import pinyin, Style


try:
    from Levenshtein import distance as calc_levenshtein_distance
except ImportError:
    def calc_levenshtein_distance(s1: str, s2: str):
        len_s1, len_s2 = len(s1), len(s2)
        dp = np.zeros((len_s1 + 1, len_s2 + 1), dtype=int)

        for i in range(len_s1 + 1):
            dp[i][0] = i
        for j in range(len_s2 + 1):
            dp[0][j] = j

        for i in range(1, len_s1 + 1):
            for j in range(1, len_s2 + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    res = dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 2
                    dp[i][j] = np.min(res)

        return dp[len_s1][len_s2]


def convert_chinese_to_pinyin(text: str):
    pinyin_list = pinyin(text, style=Style.TONE3, strict=False)
    pinyin_str = ''.join([item[0] for item in pinyin_list])
    return pinyin_str


def char_pinyins(text: str):
    for char in text:
        pinyin = convert_chinese_to_pinyin(char)
        yield char, pinyin


_Item = TypeVar('_Item')


def generate_n_grams(tokens: Sequence[_Item], n: int) -> Iterator[Tuple[_Item, ...]]:
    """
    Generate n-grams from a sequence of tokens.

    :param tokens: A sequence of tokens (words).
    :param n: The number of tokens in each n-gram.
    :return: An iterator over n-grams, each represented as a tuple of strings.
    """
    return (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def custom_score(query: str, choice: str, max_distance: int = 3) -> int:
    distance = calc_levenshtein_distance(query, choice)
    if distance > max_distance:
        return 0
    max_len = max(len(query), len(choice))
    score = 100 - (distance / max_len * 100)
    return int(score)


def filter_results(results):
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    filtered_results = {}

    for i, (current_word, current_score) in enumerate(sorted_results):
        is_substring_or_superset = False
        for j in range(i):
            higher_word, higher_score = sorted_results[j]
            if current_word in higher_word or higher_word in current_word:
                is_substring_or_superset = True
                break

        if not is_substring_or_superset:
            filtered_results[current_word] = current_score

    return filtered_results


class Corrector:
    def __init__(self):
        self.grams = defaultdict(lambda: defaultdict(int))
        self.gram2pinyin = {}

    def add_text(self, text: str, ns=(2, 3, 4)) -> None:
        text = text.lower()
        chars = list(char_pinyins(text))

        for n in ns:
            for gram in generate_n_grams(chars, n):
                words, pinyins = zip(*gram)
                gram = "".join(words)
                pinyin = "".join(pinyins)
                self.grams[pinyin][gram] += 1

    def confusion_set(self, word: str, limits=3) -> Dict[str, Tuple[int, int]]:
        confusion_set = {}
        query_pinyin = convert_chinese_to_pinyin(word)
        confusion_set.update({k: 100 for k in self.grams.get(query_pinyin, {}).keys()})
        res = process.extract(query=query_pinyin, choices=self.grams.keys(), limit=limits, scorer=custom_score)
        # print(f"query_pinyin: {query_pinyin}")
        for similar_pinyin, score in res:
            confusion_set.update({word: (self.grams[similar_pinyin][word], score) for word in self.grams[similar_pinyin]})
        confusion_set = filter_results(confusion_set)
        confusion_set = {k: v for k, v in confusion_set.items() if v[1] > 0 and k != word}
        confusion_set = OrderedDict(sorted(confusion_set.items(), key=lambda item: item[1], reverse=True))
        return confusion_set


if __name__ == "__main__":
    corpus = [
        "模糊搜索技术",
        "分词算法",
        "自然语言处理",
        "模糊飕锁",
        "模糊搜索阿",
        "莫湖搜索引擎",
        "词语相似度"
    ]

    query = "模糊搜索"

    # query_pinyin: mo2hu2sou1suo3
    # match result: {'模糊搜索': 100, '模糊飕锁': 100, '莫湖搜索': 92}

    corrector = Corrector()
    for text in corpus:
        corrector.add_text(text)
    print(corrector.confusion_set(query))

    # query_pinyin: mo2hu2sou1suo3
    # {'模糊搜索': 100, '模糊飕锁': 100, '莫湖搜索': 92}
