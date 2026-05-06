import random
from enum import Enum

class Basis(Enum):
    RECTILINEAR = "+" 
    DIAGONAL = "x"

class PRNG:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def seed(self, val):
        self.rng.seed(val)

    def get_bit(self) -> int: 
        return self.rng.choice([0, 1])
    
    def get_basis(self) -> Basis: 
        return self.rng.choice(list(Basis))
    
    def random(self) -> float: 
        return self.rng.random()
    
    def shuffle(self, x: list): 
        self.rng.shuffle(x)
    
    def choice(self, options: list): 
        return self.rng.choice(options)
    
    def sample(self, population, k) -> list:
        return self.rng.sample(population, k)
        
    def randint(self, a, b):
        return self.rng.randint(a, b)
