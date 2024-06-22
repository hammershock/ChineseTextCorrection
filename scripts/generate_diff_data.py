import difflib
import json
import re
from collections import defaultdict

from wheel.cli.tags import tags


def extract_content_after_tag(text: str, tag: str = '<begin>') -> str:
    pattern = re.escape(tag) + r'(.*)'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    else:
        return ''


def extract_content_between_tags(text: str, start_tag: str, end_tag: str) -> str:
    pattern = re.escape(start_tag) + r'(.*?)' + re.escape(end_tag)
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ''


def remove_content_between_tags(text: str, start_tag: str, end_tag: str) -> str:
    pattern = re.escape(start_tag) + r'(.*?)' + re.escape(end_tag)
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text


if __name__ == '__main__':
    processed_data = "./data.tsv"
    data_path = "./raw/text_cut.json"
    output_data = "./diff.tsv"

    with open(data_path, "r") as f:
        data = json.load(f)

    tag_counts = defaultdict(int)
    situation_counts = defaultdict(int)
    total_chars = 0
    with open(processed_data, "r") as f, open(output_data, "w") as f_out:
        for line in f:
            line = line.strip()
            bvid, corrected = line.split("\t")
            corrected = corrected.replace("[verified]", "")

            original = data[bvid]
            if "[discarded]" in original:
                continue
            if "<begin>" in original:
                original = extract_content_after_tag(original)
            if "<ad_begin>" in original:
                original = remove_content_between_tags(original, f"<ad_begin>", f"<ad_end>")

            f_out.write(f"{original}\t{corrected}\n")
            matcher = difflib.SequenceMatcher(None, original, corrected)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    continue
                if tag == "replace":
                    l1 = len(original[i1:i2])
                    l2 = len(corrected[j1:j2])
                    if l1 == l2:
                        situation_counts['equal'] += 1
                    elif l1 > l2:
                        situation_counts['less'] += 1
                    elif l2 > l1:
                        situation_counts['more'] += 1
                tag_counts[tag] += 1
                print(f"{tag}: {original[i1:i2]} -> {corrected[j1:j2]}")
            total_chars += len(original[i1:i2])

    # {'insert': 420, 'replace': 35543, 'delete': 184}
    print(total_chars)
    print(tag_counts)
    print(situation_counts)
    # {'insert': 420, 'replace': 35543, 'delete': 184}
