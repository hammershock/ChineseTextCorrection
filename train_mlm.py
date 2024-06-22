import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from transformers import get_scheduler
import os


class MLMChineseDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(file_path, 'r', encoding='utf-8') as file:
            self.lines = [line.strip() for line in file if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        encoding = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        inputs = encoding['input_ids'].squeeze(0)  # Remove batch dimension
        labels = inputs.clone()

        # Applying the mask
        rand = torch.rand(inputs.shape)
        mask_arr = (rand < 0.15) * (inputs != 101) * (inputs != 102) * (inputs != 0)
        labels[~mask_arr] = -100  # We only compute loss on masked tokens

        # Masking 15% of the input tokens
        for i in range(inputs.shape[0]):
            if mask_arr[i]:
                inputs[i] = 103  # Bert mask token id

        return inputs, encoding['attention_mask'].squeeze(0), labels


def train(model, optimizer, lr_scheduler, data_loader, epochs, device, log_path, log_every, save_dir):
    model.train()
    total_loss = 0.0
    cnt = 0
    for epoch in range(epochs):
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(tqdm(data_loader, f"{epoch}/{epochs}")):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            total_loss += loss.item()
            cnt += 1
            if cnt % log_every == 0:
                avg_loss = total_loss / cnt
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, 'a') as f:
                    f.write(f'Epoch {epoch}, Step {batch_idx + 1}, Avg Loss: {avg_loss}\n')
                    f.flush()
                total_loss = 0
                cnt = 0

        model_save_path = os.path.join(save_dir, f'model_epoch_{epoch}')
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(model_save_path)
        print(f'Model saved at {model_save_path} after Epoch {epoch}')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--model', type=str, default="./bert-base-chinese")
    parser.add_argument('--save_dir', type=str, default='./output/ckpt/mlm')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_file', type=str, default="./output/log/log_mlm.txt")
    parser.add_argument('--log_every', type=int, default=50)
    return parser.parse_args()


if __name__ == '__main__':
    # Load tokenizer and model
    args = parse_arguments()

    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = BertForMaskedLM.from_pretrained(args.model).to(args.device)

    # Prepare dataset and dataloader
    dataset = MLMChineseDataset(args.data_file, tokenizer)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    num_training_steps = args.epochs * len(data_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    train(model, optimizer, lr_scheduler, data_loader, args.epochs, args.device, args.log_file, args.log_every, args.save_dir)
