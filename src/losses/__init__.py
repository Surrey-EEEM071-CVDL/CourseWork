# Copyright (c) EEEM071, University of Surrey

from .cross_entropy_loss import CrossEntropyLoss
from .hard_mine_triplet_loss import TripletLoss


def DeepSupervision(criterion, xs, y):
    """
    Args:
    - criterion: loss function
    - xs: tuple of inputs
    - y: ground truth
    """
    loss = 0.0
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss
