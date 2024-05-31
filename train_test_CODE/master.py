import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import random
import math
tfk = tf.keras
tfkl = tf.keras.layers
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)
import gc
import contain 




X = contain.get_data('v1')
y = contain.get_label('v1')


leng = int((2*len(X))/3)

X_train = X[:leng]
y_train = y[:leng]

X_test = X[leng:]
y_test = y[leng:]



# model = contain.get_Thick_VGG_model(X_test,y_test)
model = contain.get_resnet50_model(X_test,y_test)
# model = contain.get_Custom_model(X_test,y_test)



# Train the model
history = model.fit(
    x = X_train,
    y = y_train,
    batch_size = 2,
    epochs = 50,
    validation_data=(X_test, y_test),
    callbacks = [
        tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True),
        tfk.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience=3, factor=0.1, min_lr=1e-15)
    ]
).history


contain.plot_and_save(model,history,'RES')


# v2 = contain.get_images('v2')
# v3 = contain.get_images('v3')
# v4 = contain.get_images('v4')


# print(v4.shape)
# len_tot = len(v2) +len(v3) +len(v4)
# print("len traim: " ,len_tot )

# print("len test: " ,len(contain.get_images('v1')) )




# X_test = contain.get_data('v1')
# y_test = contain.get_label('v1')

# model = contain.get_VGG_model(X_test,y_test)
# model = contain.get_resnet50_model(X_test, y_test)
# model = contain.get_Custom_model(X_test, y_test)


# folders = ['v2' ,'v3', 'v4','v5', 'v15', 'v25']
# # folders = ['v15']
# stacked_all_history=[]

# for folder in folders:

#     X_train = contain.get_data(folder)
#     y_train = contain.get_label(folder)


#     # Train the model
#     history = model.fit(
#         x = X_train,
#         y = y_train,
#         batch_size = 5,
#         epochs = 50,
#         validation_data=(X_test, y_test),
#         callbacks = [
#             tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True),
#             tfk.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience=3, factor=0.1, min_lr=1e-15)
#         ]
#     ).history
#     gc.collect()
    



# contain.plot_and_save(model,stacked_all_history,'ALL_resnet_GIANT')


# print('Done Training!!')





