from .list import *

from typing import Callable, List

class AttributeName:
    def __init__(
        self, 
        name
    ):
        self.name    = name
        self.index   = -1
        self.indexed = False
    
    def __getitem__(self, hash):
        if hash is Ellipsis:
            self.indexed = True
        elif hash is int:
            self.indexed = True
            self.index = hash
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