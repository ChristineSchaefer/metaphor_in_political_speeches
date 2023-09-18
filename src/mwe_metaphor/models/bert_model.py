import torch
import torch.nn as nn
from transformers import AutoModel

from src.mwe_metaphor.models.gnc_model import ABGCN


class BertWithGCNAndMWE(nn.Module):

    def __init__(self, max_len, config, heads, heads_mwe, dropout, num_labels=2):
        super(BertWithGCNAndMWE, self).__init__()
        self.num_labels = num_labels
        self.max_len = max_len
        self.heads = heads
        self.heads_mwe = heads_mwe
        self.bert = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")
        self.gcn1 = ABGCN(config.hidden_size, config.hidden_size, heads, 2, alpha=0.1,
                          beta=0.8)  # GCN(config.hidden_size,config.hidden_size,2)
        self.gcn2 = ABGCN(config.hidden_size, config.hidden_size, heads_mwe, 2, alpha=0.3,
                          beta=0.8)  # GCN(config.hidden_size,config.hidden_size,2)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(2 * config.hidden_size, 256)
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, input_ids, target_token_idx, attention_mask, adj, adj_mwe, batch, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)  # pooled [batch, output_dim]
        gcn1 = self.gcn1(output.last_hidden_state, adj, self.heads)  # gcn.shape: [batch, max_len, output_dim]
        gcn2 = self.gcn2(output.last_hidden_state, adj_mwe, self.heads_mwe)

        gcn = torch.cat([gcn1, gcn2], dim=2)

        target_token_idx_for_gather = target_token_idx.reshape(-1, 1, 1)
        target_token_idx_for_gather = target_token_idx_for_gather.expand(-1, 1, gcn.shape[-1])
        gcn_pooled = torch.gather(gcn, 1, target_token_idx_for_gather).view(batch, -1)

        output = self.dropout(self.linear(gcn_pooled.clone().to(torch.float32)))
        logits = self.classifier(output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(input=logits.view(-1, self.num_labels).cpu(), target=labels.clone().to(torch.long).view(-1).cpu())
            return loss
        else:
            return nn.functional.log_softmax(logits, dim=1)

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = True
