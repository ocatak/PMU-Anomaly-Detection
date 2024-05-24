#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:13:49 2023

@author: ozgur
"""

from util_defdistill import Distiller

from tensorflow.keras.callbacks import EarlyStopping

import keras
import numpy as np
import pandas as pd
from tensorflow.keras.losses import mse
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

import seaborn as sns # For Data Visualization
import matplotlib.pyplot as plt # For Data Visualization

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method
from cleverhans.tf2.attacks.madry_et_al import madry_et_al
from cleverhans.tf2.attacks.spsa import spsa
from plot_keras_history import plot_history

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
#models = ['cLSTM_modellr001_noiseC','CNN_model_angleangle','LSTM_model']
models = ['CNN_model_angleangle']

attacks = ['bim','mim','pgd','madry']

max_dist = X_train.max() - X_train.min()
eps_vals = np.round([0.1*max_dist, 0.2*max_dist, 0.3*max_dist, 0.4*max_dist, 0.5*max_dist, 0.6*max_dist, 0.7*max_dist, 0.8*max_dist, 0.9*max_dist, 1.0*max_dist],2)

params_list = list(product(eps_vals,attacks))
random.shuffle(params_list)


for model_name in models:
    teacher_model = keras.models.load_model(model_name)
    student_model = keras.models.load_model(model_name)
    
    distiller = Distiller(student=student_model, teacher=teacher_model)
    loss_fn = keras.losses.MeanSquaredError()
    
    distiller.compile(optimizer='rmsprop',
                    metrics=['mse'],
                    student_loss_fn=loss_fn,
                    distillation_loss_fn=keras.losses.KLDivergence(),
                    alpha=0.1,
                    temperature=20)

    es = EarlyStopping(monitor='val_student_loss', 
                       patience=20, 
                       #min_delta=0.000001,
                       verbose=1,
                       restore_best_weights=True,
                       mode='min')
    
    hist_distill = distiller.fit(X_train, y_train, 
                                epochs=5000,
                                verbose=1,
                                callbacks=[es],
                                batch_size=1000,
                                validation_split=0.33, shuffle= True)
    plot_history(hist_distill.history)
    plt.show()
    
    X_train_pred = distiller.student.predict(X_train, verbose=0, batch_size=5000)
    
    X_test_pred = distiller.student.predict(X_test, verbose=0, batch_size=5000)
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
    plt.savefig("img/mitigation-student-" + model_name + "-pred-comparision.pdf",bbox_inches='tight')
    plt.show()
    
    X_train_pred = teacher_model.predict(X_train, verbose=0, batch_size=5000)
    
    X_test_pred = teacher_model.predict(X_test, verbose=0, batch_size=5000)
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
    plt.savefig("img/mitigation-teacher-" + model_name + "-pred-comparision.pdf",bbox_inches='tight')
    plt.show()
    
    ##################
    y_test = y_test.values.ravel()
    model_copy = tf.keras.models.clone_model(distiller.student)
    
    for layer in model_copy.layers:
        	layer.trainable = False
    model_copy.add(Dense(2, activation='softmax', name='dummy-dense'))
    model_copy.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    y_dummy = np.zeros((X_train.shape[0],2))
    mid_range = int(X_train.shape[0] / 2.0)
    y_dummy[0:mid_range,0]= 1.0
    model_copy.fit(X_train, y_dummy, epochs=5, verbose=0,batch_size=1000,
                  validation_split=0.33, shuffle= True)
    ##################
    NUMBER_OF_MALICIOUS_INPUTS = 2000

    logits_model = tf.keras.Model(model_copy.input, model_copy.output)

    for eps_val, attack_name in tqdm(params_list):
        print(eps_val, attack_name)
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

        X_test_pred = teacher_model.predict(X_test[:NUMBER_OF_MALICIOUS_INPUTS], verbose=0, batch_size=2000)
        X_test_pred_adv = teacher_model.predict(x_adv, verbose=0, batch_size=2000)
        X_test_pred_adv_student = distiller.student.predict(x_adv, verbose=0, batch_size=2000)

        malicous_df = pd.DataFrame({'X_test_pred_adv':X_test_pred_adv.reshape((NUMBER_OF_MALICIOUS_INPUTS,))})
        malicous_df['y_test'] = y_test[:NUMBER_OF_MALICIOUS_INPUTS].reshape((NUMBER_OF_MALICIOUS_INPUTS,))
        malicous_df['X_test_pred'] = X_test_pred[:NUMBER_OF_MALICIOUS_INPUTS].reshape((NUMBER_OF_MALICIOUS_INPUTS,))
        malicous_df['X_test_pred_adv_student'] = X_test_pred_adv_student[:NUMBER_OF_MALICIOUS_INPUTS].reshape((NUMBER_OF_MALICIOUS_INPUTS,))
        
        malicous_df.plot()
        plt.grid()
        plt.title(attack_name + '-' + str(eps_val))
        plt.show()


        test_mse_loss = mse(y_test, X_test_pred).numpy()
        adv_mse_loss = mse(y_test[0:NUMBER_OF_MALICIOUS_INPUTS], X_test_pred_adv).numpy()
        adv_mse_loss_student = mse(y_test[0:NUMBER_OF_MALICIOUS_INPUTS], X_test_pred_adv_student).numpy()
        
        
        df_tmp = pd.DataFrame({'model':[model_name],'attack_name':[attack_name],
                               'eps_val':[eps_val], 'mse_normal':[np.mean(test_mse_loss)],
                               'mse_adv':[np.mean(adv_mse_loss)],
                               'mse_adv_student':[np.mean(adv_mse_loss_student)]})
        
        df_tmp.to_csv('mitigation-results.csv', mode='a', header=False, index=True)