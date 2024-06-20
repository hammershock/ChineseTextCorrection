# Chinese-Text-Correction

A two-staged Chinese text correction system, based on Soft-Mask BERT models
本项目是一个基于 PyTorch 的中文文本纠错系统，利用Soft-Mask BERT模型进行中文文本的错误检测和纠正。

## 安装依赖

```bash
pip install torch transformers tqdm numpy pypinyin pyyaml joblib
```

## 使用方法

### 训练模型
训练模型需要数据文件，格式为tsv，第一列原始文本，第二列为纠正错误后的文本

```bash
python train.py --data_config ./config/data.yaml --model_config ./config/model.yaml --epochs 10 --batch_size 290
```

- `--data_config`: 数据配置文件路径，默认为 `./config/data.yaml`。
- `--model_config`: 模型配置文件路径，默认为 `./config/model.yaml`。
- `--epochs`: 训练的轮数，默认为 10。
- `--batch_size`: 每批次的样本数量，默认为 290。
- `--num_workers`: 数据加载时使用的线程数，默认为 14。
- `--lr`: 学习率，默认为 1e-5。
- `--device`: 设备类型，默认为 `cuda`。
- `--save_every`: 每隔多少轮保存一次模型，默认为 1。
- `--log_path`: 训练日志保存路径，默认为 `output/log/log.txt`。
- `--save_dir`: 模型检查点保存路径，默认为 `output/ckpt`。
- `--resume`: 从指定检查点恢复训练，默认为 `None`。

### 模型推理

```bash
python inference.py --text "这个光灵坦克射程比较长" --model_path "./output/ckpt/0.pth" --pinyin_vocab_path "pinyin_vocab.json"
```

- `--data_config`: 数据配置文件路径，默认为 `./config/data.yaml`。
- `--model_config`: 模型配置文件路径，默认为 `./config/model.yaml`。
- `--device`: 设备类型，默认为 `cuda`。
- `--text`: 待纠错的输入文本，默认为 `这个光灵坦克射程比较长`。
- `--pinyin_vocab_path`: 拼音词汇表路径，默认为 `pinyin_vocab.json`。
- `--model_path`: 训练好的模型路径，默认为 `./output/ckpt/0.pth`。

此命令将输出纠正后的文本以及每个字符的错误概率。
```text
[0.16167694 0.18457742 0.11173134 0.1624171  0.26110426 0.17090733
 0.22880052 0.21512051 0.26683483 0.07345239 0.11701562 0.1284751
 0.08035028]
 这 个 光 棱 坦 克 射 程 比 较 长 

```
