import torch
import torch.nn as nn


class LIFT(nn.Module):
    def __init__(self, ecfp_dim=1024, chemberta_dim=384, molclr_dim=512,
            extra_dim=10, hidden_dim=256, num_heads=8, 
            task_type='regression', num_classes=1):
        """
        task_type: 'regression' or 'classification'
        num_classes: used only when task_type='classification'
        """
        super(LIFT, self).__init__()
        self.task_type = task_type
        
        # Projection layers
        self.ecfp_proj = nn.Linear(ecfp_dim, hidden_dim)
        self.chemberta_proj = nn.Linear(chemberta_dim, hidden_dim)
        self.molclr_proj = nn.Linear(molclr_dim, hidden_dim)

        # Transformer encoder for feature fusion
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers=2
        )
        
        # Attention mechanism
        self.attn_layer = nn.Linear(hidden_dim, 1)

        self.fc1 = nn.Linear(hidden_dim + extra_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        # Output layer
        if self.task_type == 'regression':
            self.output_layer = nn.Linear(256, 1)
        elif self.task_type == 'classification':
            self.output_layer = nn.Linear(256, num_classes)
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")


    def forward(self, ecfp_feature, chemberta_feature, molclr_feature, extra_feature):
        
        ecfp_emb = self.ecfp_proj(ecfp_feature)
        chemberta_emb = self.chemberta_proj(chemberta_feature)
        molclr_emb = self.molclr_proj(molclr_feature)
        
        fused_feature = torch.stack([ecfp_emb, chemberta_emb, molclr_emb], dim=1)
        fused_feature = self.transformer(fused_feature)

        # Compute attention weights and weighted sum
        attn_scores = self.attn_layer(fused_feature).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        fused_feature = torch.sum(fused_feature * attn_weights, dim=1)

        # Concatenate with extra experimental features
        combined_feature = torch.cat([fused_feature, extra_feature], dim=1)

        # Fully connected layers
        x = self.fc1(combined_feature)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        output = self.output_layer(x)
        attn_weights = attn_weights.squeeze(-1)

        if self.task_type == 'regression':
            output = output.squeeze(1)

        return output, attn_weights