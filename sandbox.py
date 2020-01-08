import torch
import pandas as pd
import numpy as np
import pickle

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from models import Convi
from sklearn.externals import joblib



# # save
# embeddings_dict = {}
# with open("glove.6B.50d.txt", 'r', encoding="utf-8") as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         vector = np.asarray(values[1:], "float32")
#         embeddings_dict[word] = vector





# with open('small_embeddings_dict.p', 'wb') as file:
#     joblib.dump(embeddings_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
#     print('Stored embeddings dict...')



# load
with open('small_embeddings_dict.p', 'rb') as file:
    embeddings_dict = joblib.load(file)

print(embeddings_dict['this'])

