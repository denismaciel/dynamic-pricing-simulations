from abc import ABC, abstractmethod

Quantity = int
Price = float


class Market(ABC):
    @abstractmethod
    def realize_demand(self, p: Price) -> Quantity:
        ...
