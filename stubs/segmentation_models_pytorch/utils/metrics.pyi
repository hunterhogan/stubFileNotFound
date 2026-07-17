from . import base as base
from ..base.modules import Activation as Activation
from _typeshed import Incomplete

class IoU(base.Metric):
    eps: Incomplete
    threshold: Incomplete
    activation: Incomplete
    ignore_channels: Incomplete
    def __init__(self, eps: float = 1e-07, threshold: float = 0.5, activation=None, ignore_channels=None, **kwargs) -> None: ...
    def forward(self, y_pr, y_gt): ...

class Fscore(base.Metric):
    eps: Incomplete
    beta: Incomplete
    threshold: Incomplete
    activation: Incomplete
    ignore_channels: Incomplete
    def __init__(self, beta: int = 1, eps: float = 1e-07, threshold: float = 0.5, activation=None, ignore_channels=None, **kwargs) -> None: ...
    def forward(self, y_pr, y_gt): ...

class Accuracy(base.Metric):
    threshold: Incomplete
    activation: Incomplete
    ignore_channels: Incomplete
    def __init__(self, threshold: float = 0.5, activation=None, ignore_channels=None, **kwargs) -> None: ...
    def forward(self, y_pr, y_gt): ...

class Recall(base.Metric):
    eps: Incomplete
    threshold: Incomplete
    activation: Incomplete
    ignore_channels: Incomplete
    def __init__(self, eps: float = 1e-07, threshold: float = 0.5, activation=None, ignore_channels=None, **kwargs) -> None: ...
    def forward(self, y_pr, y_gt): ...

class Precision(base.Metric):
    eps: Incomplete
    threshold: Incomplete
    activation: Incomplete
    ignore_channels: Incomplete
    def __init__(self, eps: float = 1e-07, threshold: float = 0.5, activation=None, ignore_channels=None, **kwargs) -> None: ...
    def forward(self, y_pr, y_gt): ...
