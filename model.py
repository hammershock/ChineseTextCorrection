import torch
import torch.nn as nn
from transformers import BertModel


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
