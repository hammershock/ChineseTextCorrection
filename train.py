import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TextCorrector
from csc_dataset import make_dataset
from utils import load_yaml


def train(model: TextCorrector, optimizer, dataloader, *, device, epochs, save_dir, log_path, save_every=1):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    loss_accumulator = defaultdict(list)
    for epoch in range(epochs):
        p_bar = tqdm(dataloader, f'Train Epoch {epoch + 1}/{epochs}')
        for batch in p_bar:
            token_ids, corrected_ids, py_token_ids, correct_mask, attn_mask = (item.to(device) for item in batch)

            optimizer.zero_grad()
            output = model.forward(token_ids, attention_mask=attn_mask, pinyin_ids=py_token_ids, target_ids=corrected_ids, labels=correct_mask)
            loss = output['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_accumulator['loss'].append(loss.item())
            loss_accumulator['detector_loss'].append(output['detector_loss'].item())
            loss_accumulator['corrector_loss'].append(output['corrector_loss'].item())
            p_bar.set_postfix(detector_loss=np.mean(loss_accumulator['detector_loss']),
                              corrector_loss=np.mean(loss_accumulator['corrector_loss']),
                              total_loss=np.mean(loss_accumulator['loss']))
        if epoch % save_every == 0:
            filename = os.path.join(save_dir, f"{epoch}.pth")
            torch.save(model.state_dict(), filename)
            with open(log_path, 'a') as f:
                losses = "\t".join(f'{k}={np.mean(v)}' for k, v in loss_accumulator.items())
                f.write(f'{epoch}\t{losses}\n')
                f.flush()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, default='./config/data.yaml')
    parser.add_argument('--model_config', type=str, default='./config/model.yaml')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=14)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--log_path', type=str, default='output/log/log.txt')
    parser.add_argument('--save_dir', type=str, default='output/ckpt')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    model = TextCorrector(**load_yaml(args.model_config)).to(args.device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=args.device))
    dataset = make_dataset(**load_yaml(args.data_config))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train(model,
          optimizer,
          dataloader,
          device=args.device,
          epochs=args.epochs,
          save_dir=args.save_dir,
          log_path=args.log_path,
          save_every=args.save_every)
