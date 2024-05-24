#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 13:17:56 2022

@author: ozgur
"""

import keras
import numpy as np
import pandas as pd
from tensorflow.keras.losses import mse
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import seaborn as sns # For Data Visualization
import matplotlib.pyplot as plt # For Data Visualization

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method
from cleverhans.tf2.attacks.madry_et_al import madry_et_al
from cleverhans.tf2.attacks.spsa import spsa

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense

from itertools import product
import random

sns.set_theme(style="ticks")

data = np.load('dataset.npz')

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']


#models = ['CNN_model_angleangle', 'biLSTM_model', 'cLSTM_modellr001_noiseC','LSTM_model']
models = ['biLSTM_model', 'cLSTM_modellr001_noiseC','LSTM_model']
models = ['LSTM_model']
attacks = ['bim','mim','pgd','madry']


max_dist = X_train.max() - X_train.min()
eps_vals = np.round([0.1*max_dist, 0.2*max_dist, 0.3*max_dist, 0.4*max_dist, 0.5*max_dist, 0.6*max_dist, 0.7*max_dist, 0.8*max_dist, 0.9*max_dist, 1.0*max_dist],2)

params_list = list(product(eps_vals,attacks))
random.shuffle(params_list)

print(params_list)

for model_name in models:
        
    model = keras.models.load_model(model_name)
    #model.summary()
    ##Evaluate model
    
    X_train_pred = model.predict(X_train, verbose=0, batch_size=5000)
    
    X_test_pred = model.predict(X_test, verbose=0, batch_size=5000)
    y_test = pd.DataFrame(y_test)
    X_test_pred = pd.DataFrame(X_test_pred)
    
    y_test = pd.DataFrame(y_test)
    plotdata = y_test.join(X_test_pred, lsuffix='_left')
    PredComp_fig = plotdata.plot(title='Prediction comparison')
    PredComp_fig.set_xlabel('Timestamp')
    #PredComp_fig.set_ylabel(test.columns[0])
    PredComp_fig.legend(['True values','Test prediction'])
    PredComp_fig = PredComp_fig.get_figure()
    plt.grid()
    plt.savefig("img/" + model_name + "-pred-comparision.pdf",bbox_inches='tight')
    plt.show()
    
    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)
    
    X_test_pred = X_test_pred.values
    y_test = y_test.values
    
    mse_vals = []
    
    for i in range(len(y_test)):
        mse_vals.append(mean_squared_error(X_test_pred[i], y_test[i]).round(3))
        
    pred_vals = pd.DataFrame({'mse':mse_vals})
    
    plt.style.use('seaborn-whitegrid')
    g = sns.histplot(pred_vals, x="mse",bins=15
                     ,  log_scale=False,  kde=True,)
    g.set_yscale('log')
    plt.grid()
    plt.savefig("img/" + model_name + "-mse-hist-normal.pdf",bbox_inches='tight')
    plt.show()
    
    ##################
    model_copy = tf.keras.models.clone_model(model)
    
    for layer in model_copy.layers:
        	layer.trainable = False
    model_copy.add(Dense(2, activation='softmax', name='tmp-dense'))
    model_copy.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    y_dummy = np.zeros((X_train.shape[0],2))
    mid_range = np.int(X_train.shape[0] / 2.0)
    y_dummy[0:mid_range,0]= 1.0
    model_copy.fit(X_train, y_dummy, epochs=5, verbose=0,batch_size=1000,
                  validation_split=0.33, shuffle= True)
    ##################
    NUMBER_OF_MALICIOUS_INPUTS = 2000

    logits_model = tf.keras.Model(model_copy.input, model_copy.output)

    for eps_val, attack_name in tqdm(params_list):
        if attack_name == "fgsm":
            x_adv = fast_gradient_method(logits_model, X_train[0:NUMBER_OF_MALICIOUS_INPUTS],
                                                   eps=eps_val/200.0, norm=np.inf,targeted=False,
                                                   clip_min=X_train.min(),
                                                   clip_max=X_train.max())
        elif attack_name == "bim":
            x_adv = basic_iterative_method(logits_model, X_test[0:NUMBER_OF_MALICIOUS_INPUTS],
                                           eps=eps_val, norm=np.inf, nb_iter=2000, 
                                           eps_iter=eps_val/10.0, targeted=False)
        elif attack_name == "pgd":
            x_adv = projected_gradient_descent(logits_model, X_test[:NUMBER_OF_MALICIOUS_INPUTS],
                                               eps=eps_val, norm=np.inf, nb_iter=2000,
                                               rand_init=True,
                                               eps_iter=eps_val/10.0, targeted=False)
        elif attack_name == "mim":
            x_adv = momentum_iterative_method(logits_model, X_test[:NUMBER_OF_MALICIOUS_INPUTS],
                                              eps=eps_val, norm=np.inf, nb_iter=2000,
                                              eps_iter=eps_val/10.0, targeted=False)
            
        elif attack_name == "madry":
            x_adv = madry_et_al(logits_model, X_test[:NUMBER_OF_MALICIOUS_INPUTS],
                                eps=eps_val, norm=np.inf, nb_iter=2000,
                                eps_iter=eps_val/10.0, targeted=False)
        elif attack_name == "spsa":
            x_adv = spsa(logits_model, tf.convert_to_tensor(X_test[:NUMBER_OF_MALICIOUS_INPUTS]),eps=eps_val,
                         y=y_test[:NUMBER_OF_MALICIOUS_INPUTS].reshape((NUMBER_OF_MALICIOUS_INPUTS,),),
                         nb_iter=500, targeted=False)

        X_test_pred_adv = model.predict(x_adv, verbose=0, batch_size=2000)

        malicous_df = pd.DataFrame({'X_test_pred_pgd':X_test_pred_adv.reshape((NUMBER_OF_MALICIOUS_INPUTS,))})
        malicous_df['y_test'] = y_test[:NUMBER_OF_MALICIOUS_INPUTS].reshape((NUMBER_OF_MALICIOUS_INPUTS,))
        malicous_df['X_test_pred'] = X_test_pred[:NUMBER_OF_MALICIOUS_INPUTS].reshape((NUMBER_OF_MALICIOUS_INPUTS,))
        malicous_df.plot()
        plt.title(attack_name + '-' + str(eps_val))
        plt.show()


        test_mse_loss = mse(y_test, X_test_pred).numpy()
        adv_mse_loss = mse(y_test[0:NUMBER_OF_MALICIOUS_INPUTS], X_test_pred_adv).numpy()
        
        df_tmp = pd.DataFrame({'model':[model_name],'attack_name':[attack_name],
                               'eps_val':[eps_val], 'mse_normal':[np.mean(test_mse_loss)],
                               'mse_adv':[np.mean(adv_mse_loss)]})
        
        df_tmp.to_csv('results.csv', mode='a', header=False, index=True)