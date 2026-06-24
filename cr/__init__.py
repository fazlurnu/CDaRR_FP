'''Conflict resolution package (functional MVP and VO resolvers).'''
from .common import ResolutionConfig
from . import mvp, vo

__all__ = [
    'ResolutionConfig',
    'mvp',
    'vo',
]
