import os
import sys
import copy
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from datetime import datetime
from utile import get_legal_moves,is_legal_move,has_tile_to_flip

BOARD_SIZE=8

if torch.cuda.is_available():
    model = torch.load("C:\\Users\\thoma\\OneDrive\\Documents\\Thomas\\ENSIM_4A\\IA\\TP_Othello\\deep_learning_main_prof\\ai_othello\\save_models_CNN\\model_2105534.pt")
    
else:
    model = torch.load("C:\\Users\\thoma\\OneDrive\\Documents\\Thomas\\ENSIM_4A\\IA\\TP_Othello\\deep_learning_main_prof\\ai_othello\\save_models_CNN\\model_2105534.pt",map_location=torch.device('cpu'))
    
model.len_inpout_seq = 1
torch.save(model, "C:\\Users\\thoma\\OneDrive\\Documents\\Thomas\\ENSIM_4A\\IA\\TP_Othello\\deep_learning_main_prof\\ai_othello\\save_models_CNN\\model_2105534_3.pt")
