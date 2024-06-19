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


import torch
import torch.nn as nn
from transformers import BertForMaskedLM

class Corrector(nn.Module):
    """MLM纠错网络"""

    def __init__(self, pretrained_model_name_or_path, num_pinyins, mask_token_id):
        super(Corrector, self).__init__()
        self.bert_mlm = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path)  # 预训练MLM
        self.pinyin_embedding = nn.Embedding(num_pinyins, self.bert_mlm.config.hidden_size)
        self.pinyin_embedding.weight.data.fill_(0)  # 初始化拼音嵌入层为0
        self.mask_embed = nn.Parameter(self.bert_mlm.bert.embeddings.word_embeddings.weight[mask_token_id].clone())
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, err_probs, pinyin_ids, target_ids=None):
        # 获取拼音嵌入
        pinyin_embed = self.pinyin_embedding(pinyin_ids)  # (batch_size, seq_len, hidden_size)

        # 获取BERT的输入嵌入
        input_ids_embed = self.bert_mlm.bert.embeddings.word_embeddings(input_ids)
        mask_embed_expanded = self.mask_embed.unsqueeze(0).unsqueeze(0).expand_as(input_ids_embed)
        fused_embed = input_ids_embed * (1 - err_probs).unsqueeze(-1) + mask_embed_expanded * err_probs.unsqueeze(-1)

        # 使用BERT模型进行前向传播
        outputs = self.bert_mlm.bert(inputs_embeds=fused_embed + pinyin_embed, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # 获取所有隐藏状态

        # 将所有隐藏状态与输入嵌入相加进行残差连接
        combined_hidden_states = hidden_states[-1]  # 使用最后一层的隐藏状态
        residual_embed = fused_embed + combined_hidden_states

        # 将残差连接后的结果输入到分类器中
        logits = self.bert_mlm.cls(residual_embed)  # (batch_size, seq_len, vocab_size)

        if target_ids is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            return {'loss': loss, 'logits': logits}


def mixed_loss(*losses, mask=None) -> torch.Tensor:
    if mask is None:
        mask = [True] * len(losses)
    losses_weighted = [(1.0 / torch.sqrt(loss + 1e-8)) * loss for loss, m in zip(losses, mask) if m]
    return sum(losses_weighted)


class TextCorrector(nn.Module):
    def __init__(self, pretrained_model_name_or_path, num_pinyins, mask_token_id):
        super(TextCorrector, self).__init__()
        self.detector = Detector(pretrained_model_name_or_path)
        self.corrector = Corrector(pretrained_model_name_or_path, num_pinyins, mask_token_id)

    def forward(self, input_ids, attention_mask, pinyin_ids, target_ids=None, labels=None):
        if labels is not None and target_ids is not None:
            detector_loss = self.detector.forward(input_ids, attention_mask, labels)['loss']
            corrector_loss = self.corrector.forward(input_ids, attention_mask, labels, pinyin_ids, target_ids)['loss']
            total_loss = mixed_loss(detector_loss, corrector_loss)
            return {'detector_loss': detector_loss, 'corrector_loss': corrector_loss, 'loss': total_loss}

        err_probs = self.detector.forward(input_ids, attention_mask)
        logits = self.corrector.forward(input_ids, attention_mask, err_probs, pinyin_ids)
        return {'logits': logits, 'err_probs': err_probs}
