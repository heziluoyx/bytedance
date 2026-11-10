"""小时桶适配的LSTM、GRU与DIN局部兴趣模型及多头注意力扩展。"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class PairSequenceDataset(Dataset):
    """按样本索引引用用户历史，不复制每行的完整用户序列。"""

    def __init__(self, arrays, indices):
        self.arrays = arrays
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        i = self.indices[position]
        a = self.arrays
        u = a["row_users"][i]
        return (
            a["x"][i],
            a["target_items"][i],
            a["target_categories"][i],
            a["sequence_items"][u],
            a["sequence_categories"][u],
            np.column_stack([a["sequence_aux"][u], a["matches"][i]]),
            a["sequence_mask"][u],
            np.float32(a["y"][i]),
        )


class PurchaseSequenceModel(nn.Module):
    """共享嵌入，小时内商品池化；DIN按候选商品计算局部兴趣权重。"""

    def __init__(
        self,
        kind,
        item_count,
        category_count,
        numeric_count,
        embedding=16,
        hidden=32,
        dropout=0.2,
        heads=2,
    ):
        super().__init__()
        self.kind = kind
        width = embedding * 2
        self.item_embedding = nn.Embedding(item_count, embedding, padding_idx=0)
        self.category_embedding = nn.Embedding(category_count, embedding, padding_idx=0)
        self.history_projection = nn.Sequential(
            nn.Linear(width + 6, width), nn.ReLU(), nn.Dropout(dropout)
        )
        if kind in ["LSTM", "GRU"]:
            recurrent = nn.LSTM if kind == "LSTM" else nn.GRU
            self.recurrent = recurrent(width, hidden, batch_first=True)
            context_width = hidden
        else:
            self.activation = nn.Sequential(
                nn.Linear(width * 4, hidden), nn.PReLU(), nn.Linear(hidden, 1)
            )
            context_width = width
            if kind == "DIN_MHA":
                self.attention = nn.MultiheadAttention(
                    width, heads, dropout=dropout, batch_first=True
                )
                self.normalization = nn.LayerNorm(width)
        self.classifier = nn.Sequential(
            nn.Linear(context_width + width + numeric_count, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        numeric,
        target_item,
        target_category,
        history_item,
        history_category,
        auxiliary,
        mask,
    ):
        item_mask = history_item.ne(0).unsqueeze(-1)
        raw = torch.cat(
            [
                self.item_embedding(history_item),
                self.category_embedding(history_category),
            ],
            dim=-1,
        )
        pooled = (raw * item_mask).sum(2) / item_mask.sum(2).clamp(min=1)
        history = self.history_projection(
            torch.cat([pooled, auxiliary.float()], dim=-1)
        )
        target = torch.cat(
            [
                self.item_embedding(target_item),
                self.category_embedding(target_category),
            ],
            dim=-1,
        )
        if self.kind in ["LSTM", "GRU"]:
            # 右侧填充时直接取最后有效输出，保持先后语义并使用CPU融合内核。
            lengths = mask.sum(1).clamp(min=1)
            outputs, _ = self.recurrent(history)
            context = outputs[torch.arange(len(outputs)), lengths - 1]
            context = context * mask.any(1).unsqueeze(1)
        else:
            if self.kind == "DIN_MHA":
                safe = mask.clone()
                safe[~safe.any(1), 0] = True
                attended, _ = self.attention(
                    history,
                    history,
                    history,
                    key_padding_mask=~safe,
                    need_weights=False,
                )
                history = self.normalization(history + attended)
            query = target.unsqueeze(1).expand_as(history)
            scores = self.activation(
                torch.cat([query, history, query - history, query * history], dim=-1)
            ).squeeze(-1)
            # DIN局部激活保留兴趣强度，不把同小时商品强制排成事件序列。
            weights = torch.sigmoid(scores) * mask
            context = (history * weights.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(
                min=1
            ).sqrt().unsqueeze(1)
        return self.classifier(torch.cat([context, target, numeric], dim=-1)).squeeze(
            -1
        )
