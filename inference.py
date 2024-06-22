import argparse

import torch
from transformers import BertTokenizer

from ctc_dataset import pinyin_tokenize
from model import TextCorrector
from utils import load_yaml, load_json


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, default='./config/data.yaml')
    parser.add_argument('--model_config', type=str, default='./config/model.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--text', type=str, default='先出一个光灵坦克,用光棱坦克搓他')
    parser.add_argument('--pinyin_vocab_path', type=str, default='pinyin_vocab.json')
    parser.add_argument('--model_path', type=str, default="./output/ckpt/6.pth")
    return parser.parse_args()


def inference(texts, err_threshold=0.8):
    result = tokenizer(texts, return_tensors="pt", truncation=True, max_length=128)

    tokens, py_tokens = pinyin_tokenize(texts, tokenizer)
    py_tokens = ['[CLS]'] + list(py_tokens) + ['[SEP]']
    # tokens = ['[CLS]'] + list(tokens) + ['[SEP]']
    py_token_ids = [pinyin_vocab[item] for item in py_tokens]
    input_ids = result['input_ids'].to(args.device)
    attention_mask = result['attention_mask'].to(args.device)
    pinyin_ids = torch.LongTensor(py_token_ids).to(args.device).unsqueeze(0)
    with torch.no_grad():
        output = model.forward(input_ids, attention_mask=attention_mask, pinyin_ids=pinyin_ids)
        logits = output['logits']
        # topk_values, topk_indices = torch.topk(logits, 5, dim=-1)
        err_probs = output['err_probs']
        output_ids = input_ids.clone()
        preds = torch.argmax(logits, dim=-1)
        mask = err_probs > err_threshold

        output_ids[mask] = preds[mask]

        return output_ids.cpu().numpy(), output['err_probs'].cpu().numpy()


if __name__ == '__main__':
    args = parse_arguments()
    config = load_yaml(args.model_config)
    model = TextCorrector(**config).to(args.device)
    model.eval()
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    tokenizer = BertTokenizer.from_pretrained(config["pretrained_model_name_or_path"])
    pinyin_vocab = load_json(args.pinyin_vocab_path)
    results, err_probs = inference(args.text)
    print(tokenizer.decode(results[0]))
    print(err_probs)
