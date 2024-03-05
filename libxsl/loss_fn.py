import torch
import torch.nn as nn

class LRSQLoss(nn.Module):
    def __init__(self, omega):
        super(LRSQLoss, self).__init__()
        self.omega = omega
        self.amp = 20

    def forward(self, predictions, labels):
        # Apply logistic loss for label = 1
        logistic_loss = labels*F.binary_cross_entropy_with_logits(self.amp*predictions, labels, reduction='none')

        # Apply squared loss for label = 0
        squared_loss = (1-labels)*self.omega*(predictions + 1.0)**2


        loss = logistic_loss + squared_loss



        # Select the appropriate loss based on the label
        #loss = torch.where(labels == 1, logistic_loss, squared_loss)

        return loss.mean()

class LRL2Loss(nn.Module):
    def __init__(self, omega):
        super(LRL2Loss, self).__init__()
        self.omega = omega

    def forward(self, predictions, labels):
        # Apply logistic loss for label = 1
        logistic_loss = labels*F.binary_cross_entropy_with_logits(predictions, labels, reduction='none')

        # Apply squared loss for label = 0
        l2_loss = (1-labels)*self.omega*(torch.max(torch.zeros_like(predictions), predictions + 1.0))**2


        loss = logistic_loss + l2_loss

        return loss.mean()

class LRExpLoss(nn.Module):
    def __init__(self, omega):
        super(LRExpLoss, self).__init__()
        self.omega = omega

    def forward(self, predictions, labels):
        # Apply logistic loss for label = 1
        logistic_loss = labels*F.binary_cross_entropy_with_logits(predictions, labels, reduction='none')

        # Apply squared loss for label = 0
        exp_loss = (1-labels)*self.omega*torch.exp(predictions)


        loss = logistic_loss + exp_loss

        return loss.mean()

class ExpLoss(nn.Module):
    def __init__(self, gamma=20):
        super(ExpLoss, self).__init__()
        self.gamma = gamma

    def forward(self, predictions, labels):

        loss = torch.exp((1-2*labels)*predictions*self.gamma)

        return loss.mean()

class SQLoss(nn.Module):
    def __init__(self, omega):
        super(SQLoss, self).__init__()
        self.omega = omega

    def forward(self, predictions, labels):

        pos_loss = labels * (predictions - 1.0) ** 2

        # Apply squared loss for label = 0
        neg_loss = (1-labels)*self.omega*(predictions + 1.0)**2

        loss = pos_loss + neg_loss

        return loss.mean()

class L2SQLoss(nn.Module):
    def __init__(self, omega):
        super(L2SQLoss, self).__init__()
        self.omega = omega

    def forward(self, predictions, labels):

        #pos_loss = labels*(torch.max(torch.zeros_like(predictions), 1-20*predictions))**2
        #pos_loss = labels*torch.exp(-25*predictions)

        # Apply squared loss for label = 0
        #neg_loss = (1-labels)*self.omega*(predictions + 1.0)**2
        #neg_loss = (1-labels)*(torch.max(torch.zeros_like(predictions), 20*predictions + 1))**2
        #neg_loss = (1-labels)*F.binary_cross_entropy_with_logits(predictions*5, labels, reduction='none')
        loss = F.binary_cross_entropy_with_logits(predictions*10, labels, reduction='none')

        #loss = pos_loss + neg_loss

        return loss.mean()

class LogSoftmaxLoss(nn.Module):
    def __init__(self, omega, kernel_approx=False):
        super(LogSoftmaxLoss, self).__init__()
        self.omega = omega
        self.kernel_approx = kernel_approx

    def forward(self, predictions, labels):
        if self.kernel_approx:
            exp_pred = predictions
        else:
            exp_pred = torch.exp(predictions*self.omega)
        exp_sum = (exp_pred*(1-labels)).sum(dim=-1)
        non_zeros = labels.nonzero()
        rows = non_zeros[:, 0]
        cols = non_zeros[:, 1]
        loss = -torch.log(exp_pred[rows, cols] / (exp_sum[rows] + exp_pred[rows, cols]))
        return loss.mean()


