from .base import KnowledgeTracingModel
from .bkt import StandardBKT, BKTWithForgetting, IndividualizedBKT, ImprovedBKT
from .logistic import LogisticModel
# Deep model import is conditional based on torch availability (handled in client code or here)
# To keep it simple, we don't import DKT here to avoiding crashing if torch missing
