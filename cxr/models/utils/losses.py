import torch
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

class CornLossMulti(torch.nn.Module):
    """
    Compute the CORN loss for multi-class classification.
    """
    def __init__(self, class_labels_num):
        super().__init__()
        self.class_labels_num = class_labels_num # [Classes, Labels]
    
    def forward(self, logits, targets):
        """
        Args:
            logits: torch.Tensor, shape [batch_size, num_classes*(num_labels-1)]
            targets: torch.Tensor, shape [batch_size]
        """
        B = logits.shape[0]
        chunks = torch.split(logits, self.class_labels_num, dim=1)
        loss = 0
        for c, chunk in enumerate(chunks):
            loss += corn_loss(chunk, targets[:, c], chunk.shape[1]+1)
        return loss

    def logits2labels(self, logits):
        """
        Args:
            logits: torch.Tensor, shape [batch_size, num_classes*(num_labels-1)]
        """
        chunks = torch.split(logits, self.class_labels_num, dim=1)
        labels = []
        for c, chunk in enumerate(chunks):
            label = corn_label_from_logits(chunk)
            labels.append(label)
        return torch.stack(labels, dim=-1)