import argparse

import torch
from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup, \
    DataCollatorForLanguageModeling, BertConfig
from torch.utils.data import DataLoader

from mlm_dataset import MLMDataset
from utils import load_yaml


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, default='./config/data.yaml')
    parser.add_argument('--model_config', type=str, default='./config/model.yaml')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=14)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--log_path', type=str, default='output/log/log_pretrained_mlm.txt')
    parser.add_argument('--save_dir', type=str, default='output/ckpt/pretrained_mlm')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    model_name = load_yaml(args.model_config)["pretrained_model_name_or_path"]

    model = BertForMaskedLM.from_pretrained(model_name).to(args.device)
    # config = BertConfig.from_pretrained(model_name)
    #
    # new_max_position_embeddings = 2048
    # config.max_position_embeddings = new_max_position_embeddings
    #
    # old_position_embeddings = model.bert.embeddings.position_embeddings.weight.data
    # old_max_position_embeddings, hidden_size = old_position_embeddings.size()
    #
    # new_position_embeddings = torch.nn.functional.interpolate(
    #     old_position_embeddings.unsqueeze(0).permute(0, 2, 1),
    #     size=new_max_position_embeddings,
    #     mode='linear'
    # ).squeeze(0).permute(1, 0)
    #
    # model.bert.embeddings.position_embeddings = torch.nn.Embedding(new_max_position_embeddings, hidden_size)
    # model.bert.embeddings.position_embeddings.weight.data = new_position_embeddings
    #
    # model = BertForMaskedLM.from_pretrained(model_name).to(args.device)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    dataset = MLMDataset(**load_yaml(args.data_config)['train'])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_collator)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(data_loader) * args.epochs)

    model.train()
    for epoch in range(args.epochs):
        for batch in data_loader:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
