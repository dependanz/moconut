from .list import *

from typing import Callable, List

class AttributeName:
    def __init__(
        self, 
        name
    ):
        self.name = name
        self.indexed = False
    
    def __getitem__(self, hash):
        if hash is Ellipsis:
            self.indexed = not self.indexed
        return self

    def __str__(self):
        return f"{self.name}" + ("[...]" if self.indexed else "")

class DependentDefault:
    def __init__(
        self, 
        parents    : List[str], 
        dependence : Callable
    ):
        self.parents = parents
        self.dependence = dependence