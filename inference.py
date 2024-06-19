import argparse

import torch
from transformers import BertTokenizer

from csc_dataset import pinyin_tokenize
from model import TextCorrector
from utils import load_yaml, load_json


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, default='./config/data.yaml')
    parser.add_argument('--model_config', type=str, default='./config/model.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--text', type=str, default='好我们来看这一句美人一小快递')
    parser.add_argument('--pinyin_vocab_path', type=str, default='pinyin_vocab.json')
    parser.add_argument('--model_path', type=str, default="./output/ckpt/0(1).pth")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    config = load_yaml(args.model_config)
    model = TextCorrector(**config).to(args.device)
    model.eval()
    # model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.detector.load_state_dict(torch.load("output/ckpt/detector/4.pth"))
    tokenizer = BertTokenizer.from_pretrained(config["pretrained_model_name_or_path"])
    result = tokenizer(args.text, return_tensors="pt")
    tokens, py_tokens = pinyin_tokenize(args.text, tokenizer)
    py_tokens = ['[CLS]'] + list(py_tokens) + ['[SEP]']
    pinyin_vocab = load_json(args.pinyin_vocab_path)
    py_token_ids = [pinyin_vocab[item] for item in py_tokens]
    with torch.no_grad():
        print(result['input_ids'])
        output = model.forward(result['input_ids'].to(args.device),
                               result['attention_mask'].to(args.device),
                               torch.LongTensor(py_token_ids).to(args.device).unsqueeze(0))
        logits = output['logits']
    preds = torch.argmax(logits, dim=-1)[0]
    line = tokenizer.decode(preds.cpu().numpy())

    print(output['err_probs'].cpu().numpy()[0])
    print(line)