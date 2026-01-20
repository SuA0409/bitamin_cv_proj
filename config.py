import os

# Dataset root path
'''
    define your root directory
'''
ROOT = r"C:\Users\user\Desktop\leesua\animalcleft2025\data"

# Model settings
MEGAD_NAME = 'hf-hub:BVRA/MegaDescriptor-L-384'
EVA_NAME = 'EVA02-L-14-336'
EVA_WEIGHT_NAME = 'merged2b_s6b_b61k'
DEVICE = 'cuda'

# Threshold
THRESHOLD = 0.35