from enum import Enum

# -----------------------------------------------------------------

class BaseClass:
    def __init__(self):
        """Base class"""

        self.regularizer = None
        self.flag_set = 0

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer
        self.flag_set = 1

# -----------------------------------------------------------------

class Phase(Enum):
    """Enum class"""
    
    train = 1
    test = 2
    validation = 3

# -----------------------------------------------------------------
