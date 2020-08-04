from abc import ABC, abstractmethod


class Seller(ABC):
    """
    * Seller has an idea how Market works
    * Seller sets a price
    * Seller observes demand
    * Seller updates beliefs
    """

    def __init__(
        self, beliefs,
    ):
        self.beliefs = beliefs

    def update_beliefs(self):
        ...
