'''Conflict detection package.'''
from .common import ConflictState
from .statebased import detect

__all__ = ['ConflictState', 'detect']
