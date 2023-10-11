from abc import ABC, abstractmethod
from typing import List, Union

import torch
from torch import Tensor


class LossAggregationStrategy(ABC):
    @abstractmethod
    def add_executor_loss(self, loss: Tensor) -> None:
        pass

    @abstractmethod
    def agg_loss(self) -> Tensor:
        pass

    @abstractmethod
    def reset_batch(self):
        pass


class FEDAVGLossAggregation(LossAggregationStrategy):
    agg_tensors: List[Tensor] = []

    def add_executor_loss(self, loss: Tensor) -> None:
        self.agg_tensors.append(loss)

    def agg_loss(self) -> Tensor:
        sum_tensor: Union[Tensor, None] = None
        for i in self.agg_tensors:
            if sum_tensor is None:
                sum_tensor = i
            else:
                sum_tensor += i
        return torch.div(sum_tensor, len(self.agg_tensors))

    def reset_batch(self):
        self.agg_tensors.clear()

class FEDSUMLossAggregation(LossAggregationStrategy):
    agg_tensors: List[Tensor] = []

    def add_executor_loss(self, loss: Tensor) -> None:
        self.agg_tensors.append(loss)

    def agg_loss(self) -> Tensor:
        sum_tensor: Union[Tensor, None] = None
        for i in self.agg_tensors:
            if sum_tensor is None:
                sum_tensor = i
            else:
                sum_tensor += i
        return sum_tensor

    def reset_batch(self):
        self.agg_tensors.clear()


FED_AVG: LossAggregationStrategy = FEDAVGLossAggregation()
FED_SUM: LossAggregationStrategy = FEDSUMLossAggregation()