from collections.abc import Iterable
from typing import Sequence, TypeVar, Iterator, Tuple, List, Union

_T = TypeVar("_T")


def batchify(*sequences: Sequence[_T], max_len: int, overlap: int, pad: Union[List[_T], _T]) -> Iterator[
    Tuple[Tuple[List[_T], ...], List[int]]]:
    """batchify the sequence data"""
    assert len(set(len(seq) for seq in sequences)) == 1, "all sequences must have the same length"
    if not isinstance(pad, Iterable):
        pad = [pad] * len(sequences)  # num sequences
    assert len(pad) == len(sequences), "pad list length must match number of sequences"
    seq_len = len(sequences[0])
    # Generate slices with overlap
    slices = []
    start = 0
    while start < seq_len:
        end = min(start + max_len, seq_len)
        slices.append(slice(start, end))
        if end == seq_len:
            break
        start += max_len - overlap

    for slice_ in slices:
        segmented_seqs = tuple(seq[slice_] for seq in sequences)

        # Pad the sequences if necessary
        padded_seqs = []
        for seq, pad_value in zip(segmented_seqs, pad):
            if len(seq) < max_len:
                padded_seq = list(seq) + [pad_value] * (max_len - len(seq))
            else:
                padded_seq = list(seq)
            padded_seqs.append(padded_seq)

        pad_mask = [0] * max_len
        if len(segmented_seqs[0]) < max_len:
            pad_mask = [1 if i >= len(segmented_seqs[0]) else 0 for i in range(max_len)]

        yield tuple(padded_seqs), pad_mask


if __name__ == '__main__':
    # 测试函数
    sequences = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    ]
    max_len = 4
    overlap = 2
    pad = 0

    for batch, mask in batchify(*sequences, max_len=max_len, overlap=overlap, pad=[100, 'z']):
        print("Batch:", batch)
        print("Pad mask:", mask)
