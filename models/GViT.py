import torch
from torch import nn
from torch_geometric.nn import GCNConv
from transformers import ViTForImageClassification, ViTConfig


class GViT(nn.Module):
    def __init__(self, adj_matrix):
        super(GViT, self).__init__()

        # GNN
        edge_index = torch.tensor(adj_matrix.nonzero(), dtype=torch.long)
        edge_weight = torch.tensor(adj_matrix[adj_matrix != 0], dtype=torch.float)

        self.edge_index = edge_index
        self.edge_weight = edge_weight

        self.gnn1 = GCNConv(in_channels=500, out_channels=256)

        self.batchnorm_gnn = nn.BatchNorm1d(num_features=256)

        # ViT
        model_name = "google/vit-base-patch16-224"
        config = ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (129, 1)})
        config.update({'patch_size': (8, 1)})

        self.vit = ViTForImageClassification.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True
        )
        self.vit.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
            256, 768, kernel_size=(8, 1), stride=(8, 1), padding=(0, 0), groups=256
        )
        self.vit.classifier = nn.Sequential(
            nn.Linear(768, 1000),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1000, 2)
        )

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.squeeze(1)

        device = x.device
        self.edge_index = self.edge_index.to(device)
        self.edge_weight = self.edge_weight.to(device)

        batch_size, num_nodes, num_features = x.shape
        x = x.reshape(batch_size * num_nodes, num_features)  # Flatten 节点维度
        x = self.gnn1(x, self.edge_index, self.edge_weight)  # GNN 处理
        x = self.batchnorm_gnn(x)  # 批量归一化
        x = x.reshape(batch_size, num_nodes, -1)  # 恢复形状

        x = x.permute(0, 2, 1).unsqueeze(-1)  # 调整为 (batch_size, 256, 129, 1)
        x = self.dropout(x)

        x = self.vit(x).logits
        return x
