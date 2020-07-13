import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from .GraphSimRec import GraphRecommender
from .RecAPI import app