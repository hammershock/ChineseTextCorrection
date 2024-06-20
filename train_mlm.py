import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup, \
    DataCollatorForLanguageModeling, BertConfig
from torch.utils.data import DataLoader

from mlm_dataset import MLMDataset
from utils import load_yaml


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, default='./config/data.yaml')
    parser.add_argument('--model_config', type=str, default='./config/model.yaml')
    parser.add_argument('--epochs', type=int, default=280)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=14)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('-p', '--mlm_prob', type=float, default=0.15, help='Masked language modeling probability')
    parser.add_argument('--log_path', type=str, default='output/log/log_pretrained_mlm.txt')
    parser.add_argument('--save_dir', type=str, default='output/ckpt/pretrained_mlm')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def train(model, optimizer, dataloader, *, device, epochs, save_dir, log_path, save_every=1):
    model.train()
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    loss_accumulator = defaultdict(list)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(data_loader) * args.epochs)

    for epoch in range(epochs):
        p_bar = tqdm(dataloader, f'Train Epoch {epoch + 1}/{epochs}')
        for batch in p_bar:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)

            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            loss_accumulator['loss'].append(loss.item())
            p_bar.set_postfix(total_loss=np.mean(loss_accumulator['loss']))
        if epoch % save_every == 0:
            # filename = os.path.join(save_dir, f"{epoch}.pth")
            model.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            with open(log_path, 'a') as f:
                losses = "\t".join(f'{k}={np.mean(v)}' for k, v in loss_accumulator.items())
                f.write(f'{epoch}\t{losses}\n')
                f.flush()


if __name__ == '__main__':
    args = parse_arguments()
    model_name = load_yaml(args.model_config)["pretrained_model_name_or_path"]

    model = BertForMaskedLM.from_pretrained(model_name).to(args.device)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    dataset = MLMDataset(**load_yaml(args.data_config)['train'])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_prob,
    )

    data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_collator)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    train(model, optimizer, data_loader,
          device=args.device,
          epochs=args.epochs,
          save_dir=args.save_dir,
          log_path=args.log_path,
          save_every=args.save_every)

