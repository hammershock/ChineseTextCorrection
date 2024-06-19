import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Detector
from csc_dataset import make_dataset


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train(model, optimizer, dataloader, *, device, epochs, save_dir, log_path, save_every=1):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    loss_accumulator = []
    for epoch in range(epochs):
        for batch in tqdm(dataloader, f'Train Epoch {epoch + 1}/{epochs}'):
            token_ids, _, _, correct_mask, attn_mask = batch
            token_ids = token_ids.to(device)
            attn_mask = attn_mask.to(device)
            correct_mask = correct_mask.to(device)

            optimizer.zero_grad()
            output = model(token_ids, attention_mask=attn_mask, labels=correct_mask)
            loss = output['loss']
            loss.backward()
            optimizer.step()
            loss_accumulator.append(loss.item())
        if epoch % save_every == 0:
            filename = os.path.join(save_dir, f"{epoch}.pth")
            torch.save(model.state_dict(), filename)
            with open(log_path, 'a') as f:
                f.write(f'{epoch}\t{np.mean(loss_accumulator)}\n')
                f.flush()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, default='./config/data.yaml')
    parser.add_argument('--model_config', type=str, default='./config/model.yaml')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=14)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--log_path', type=str, default='output/log/detector.txt')
    parser.add_argument('--save_dir', type=str, default='output/ckpt/detector')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    model = Detector(**load_yaml(args.model_config)['detector']).to(args.device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=args.device))
    dataset = make_dataset(**load_yaml(args.data_config))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train(model,
          optimizer,
          dataloader,
          device=args.device,
          epochs=args.epochs,
          save_dir=args.save_dir,
          log_path=args.log_path,
          save_every=args.save_every)

