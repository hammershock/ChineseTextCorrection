import torch
import torch.nn as nn
from transformers import BertModel, BertForMaskedLM


class Detector(nn.Module):
    """错误检测网络"""

    def __init__(self, pretrained_model_name_or_path='bert-base-chinese'):
        super(Detector, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.hidden_size = self.bert.config.hidden_size
        self.linear = nn.Linear(self.hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        :param input_ids:
        :param attention_mask:
        :param labels: (FloatTensor) 某个位置的错误概率
        :return: 如果给定 labels，则返回 (loss, probabilities)，否则返回 probabilities
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        logits = self.linear(sequence_output)  # (batch_size, seq_len, 1)
        logits = logits.squeeze(-1)  # (batch_size, seq_len)
        probabilities = torch.sigmoid(logits)  # (batch_size, seq_len)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {'loss': loss, 'probabilities': probabilities}

        return probabilities


class Corrector(nn.Module):
    """MLM纠错网络"""

    def __init__(self, pretrained_model_name_or_path, num_pinyins, mask_token_id):
        super(Corrector, self).__init__()
        self.bert_mlm = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path)  # 预训练MLM
        self.pinyin_embedding = nn.Embedding(num_pinyins, self.bert_mlm.bert.config.hidden_size)
        self.mask_embed = nn.Parameter(self.bert_mlm.bert.embeddings.word_embeddings.weight[mask_token_id].clone())
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, err_probs, pinyin_ids, target_ids=None):
        # Extra pinyin embeddings
        pinyin_embed = self.pinyin_embedding(pinyin_ids)  # (batch_size, seq_len, hidden_size)

        # Get the BERT embeddings for input_ids
        input_ids_embed = self.bert_mlm.bert.embeddings.word_embeddings(input_ids)
        mask_embed_expanded = self.mask_embed.unsqueeze(0).unsqueeze(0).expand_as(input_ids_embed)
        input_ids_embed = input_ids_embed * (1 - err_probs).unsqueeze(-1) + mask_embed_expanded * err_probs.unsqueeze(-1)

        # Use BERT model with combined embeddings
        outputs = self.bert_mlm(inputs_embeds=input_ids_embed + pinyin_embed, attention_mask=attention_mask)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        if target_ids is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            return {'loss': loss, 'logits': logits}
        return logits
