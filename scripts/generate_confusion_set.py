import os

from tqdm import tqdm

from scripts.corrector import Corrector


if __name__ == '__main__':
    # build confusion set
    output_path = "confusion_set.txt"

    if os.path.exists(output_path):
        raise FileExistsError

    corrector = Corrector()
    with open("../data/raw_data_segments.txt", "r") as f:
        lines = f.readlines()
    for data, bvid in tqdm(lines):
        corrector.add_text(data)

    with open('../data/vocabulary.txt', "w") as f:
        words = [line.strip().split()[0] for line in f]
    words = [word for word in words if len(word) > 1]

    with open(output_path, "w") as f:
        for word in tqdm(words):
            confusion_set = corrector.confusion_set(word, limits=15)
            confusion_words = " ".join(word for word, score in confusion_set.items())
            f.write(f"{word}:\t{confusion_words}\n")
            f.flush()
