import torch
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

import torch.nn.functional as F

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
        chunks = torch.split(logits, self.class_labels_num, dim=1)
        loss = 0
        for c, chunk in enumerate(chunks):
            loss += corn_loss(chunk, targets[:, c], chunk.shape[1]+1)
        return loss/len(chunks)

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
    


class CELossMulti(torch.nn.Module):
    """
        Multi CE Loss  
    """
    def __init__(self, class_labels_num):
        super().__init__()
        self.class_labels_num = class_labels_num # [Classes, Labels]
    
    def forward(self, logits, targets):
        """
        Args:
            logits: torch.Tensor, shape [batch_size, num_classes*num_labels]
            targets: torch.Tensor, shape [batch_size, num_classes]
        """
        chunks = torch.split(logits, self.class_labels_num, dim=1)
        loss = 0
        for c, chunk in enumerate(chunks):
            # loss += F.cross_entropy(chunk, targets[:, c])
            loss += F.binary_cross_entropy_with_logits(chunk, F.one_hot(targets[:, c], num_classes=self.class_labels_num[c]).float())
        return loss/len(chunks)

    def logits2labels(self, logits):
        """
        Args:
            logits: torch.Tensor, shape [batch_size, num_classes*num_labels]
        """
        chunks = torch.split(logits, self.class_labels_num, dim=1)
        labels = []
        for c, chunk in enumerate(chunks):
            label = torch.argmax(chunk, dim=-1)
            labels.append(label)
        return torch.stack(labels, dim=-1)