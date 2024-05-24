#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 13:01:15 2022

@author: ozgur
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()

style_list = ['default', 'classic'] + sorted(
        style for style in plt.style.available
        if style != 'classic' and not style.startswith('_'))

def create_tables_for_attacks():
    models = ['CNN_model_angleangle', 'biLSTM_model', 'cLSTM_modellr001_noiseC','LSTM_model']
    
    df = pd.read_csv('results.csv',names=['model','attack_name','eps_val','mse_normal','mse_adv'])
    df_sorted = df.sort_values(by=['model','eps_val'])
    df_sorted = df_sorted.groupby(by=['model','attack_name','eps_val']).mean()
    
    print(df_sorted.head())
    
    
    f = open('table.txt','w')
    f.write(df_sorted.to_latex())
    f.close()
    print(df_sorted.to_latex())
    
    plt.style.use('default')
    plt.style.use(style_list[19])
    
    for model_name in models:
        tmp_df = df.query("model=='" + model_name + "'")
        #print(tmp_df.sort_values(['eps_val']))
        sns.lineplot(data=tmp_df, x="eps_val", y="mse_adv", hue="attack_name")
        plt.title(model_name)
        plt.ylim([-0.2,9])
        plt.grid()
        plt.xlabel(r'$\epsilon$-budget', fontsize=16)
        plt.ylabel(r'MSE', fontsize=16)
        L = plt.legend(loc=2,fontsize=14, title='Attacks', shadow=True) # using a size in points
        for i in range(len(L.get_texts())):
            L.get_texts()[i].set_text(L.get_texts()[i].get_text().upper())
    
        plt.setp(L.get_title(),fontsize='14')
        plt.savefig(model_name + ".pdf",bbox_inches='tight')
        plt.show()
        
        tmp_df.drop('model', inplace=True, axis=1)
        tmp_df = tmp_df.groupby(by=['attack_name','eps_val']).mean().reset_index()
        
        tmp_df =tmp_df.pivot(index ='eps_val', columns =['attack_name'], values =['mse_normal', 'mse_adv']).reset_index()
        tmp_df.columns= tmp_df.columns.map('{0[0]}-{0[1]}'.format) 
        tmp_df.drop(['mse_normal-madry','mse_normal-mim','mse_normal-pgd'], inplace=True, axis=1)
        
        f = open(model_name + '-table.txt','w')
        f.write(tmp_df.to_latex(index=False))
        f.close()
        
        
        print(tmp_df)

def create_tables_for_mitigtaion(model_name, attack_name, eps_val):
    models = ['CNN_model_angleangle', 'biLSTM_model', 'cLSTM_modellr001_noiseC','LSTM_model']
    
    df = pd.read_csv('mitigation-results.csv',names=['model','attack_name',
                                                     'eps_val','mse_normal',
                                                     'mse_adv','mse_adv_student'])
    df_sorted = df.sort_values(by=['model','eps_val'])
    df_sorted = df_sorted.groupby(by=['model','attack_name','eps_val']).mean().reset_index()
    
    
    df_sorted.drop(['model'], axis=1, inplace=True)
    print(df_sorted)
    
    f = open('mitigation-table.txt','w')
    f.write(df_sorted.to_latex())
    f.close()
    print(df_sorted.to_latex())
    
    for model_name in models:
        tmp_df = df.query("model=='" + model_name + "'")
        #print(tmp_df.sort_values(['eps_val']))
        sns.lineplot(data=tmp_df, x="eps_val", y="mse_adv", hue="attack_name")
        plt.title(model_name)
        plt.ylim([-0.2,9])
        plt.grid()
        plt.xlabel(r'$\epsilon$-budget', fontsize=16)
        plt.ylabel(r'MSE', fontsize=16)
        L = plt.legend(loc=2,fontsize=14, title='Attacks', shadow=True) # using a size in points
        for i in range(len(L.get_texts())):
            L.get_texts()[i].set_text(L.get_texts()[i].get_text().upper())
    
        plt.setp(L.get_title(),fontsize='14')
        plt.savefig(model_name + ".pdf",bbox_inches='tight')
        plt.show()
        
        tmp_df.drop('model', inplace=True, axis=1)
        tmp_df = tmp_df.groupby(by=['attack_name','eps_val']).mean().reset_index()
        
        tmp_df =tmp_df.pivot(index ='eps_val', columns =['attack_name'], values =['mse_normal', 'mse_adv','mse_adv_student']).reset_index()
        tmp_df.columns= tmp_df.columns.map('{0[0]}-{0[1]}'.format) 
        
        
        
        tmp_df.drop(['mse_normal-bim','mse_normal-madry','mse_normal-mim','mse_normal-pgd'], inplace=True, axis=1)
        
        # Define the desired column order
        desired_columns = [
            'eps_val-',
            'mse_adv-bim',
            'mse_adv_student-bim',
            'mse_adv-madry',
            'mse_adv_student-madry',
            'mse_adv-mim',
            'mse_adv_student-mim',
            'mse_adv-pgd',
            'mse_adv_student-pgd'
        ]
        
        # Reorder the columns in the DataFrame
        tmp_df = tmp_df[desired_columns]

        
        f = open(model_name + '-mitigated-table.txt','w')
        f.write(tmp_df.to_latex(index=False))
        f.close()
        
        
        print(tmp_df)
        
def create_plots_for_mitigtaion(model_name, attack_name, eps_val):
    sns.set_style("whitegrid")
    models = ['CNN_model_angleangle', 'biLSTM_model', 'cLSTM_modellr001_noiseC','LSTM_model']
    
    df = pd.read_csv('mitigation-results.csv',names=['model','attack_name',
                                                     'eps_val','mse_normal',
                                                     'mse_adv','mse_adv_student'])
    df_sorted = df.sort_values(by=['model','attack_name', 'eps_val'])
    print(df_sorted.sample(10))
    print(list(df_sorted))

    for model_name in models:
        tmp_df = df_sorted.query("model=='" + model_name + "'")
        fig, ax = plt.subplots()
        
        # Plotting Normal line
        normal_line = sns.lineplot(data=tmp_df, x="eps_val", y="mse_normal", hue="attack_name",
                                   linestyle="--", label="Normal", ax=ax, palette="gray")
        
        # Plotting Undefended line
        undefended_line = sns.lineplot(data=tmp_df, x="eps_val", y="mse_adv", hue="attack_name",
                                       linestyle="-.", label="Undefended", ax=ax, palette="ocean")
        
        # Plotting Defended line
        defended_line = sns.lineplot(data=tmp_df, x="eps_val", y="mse_adv_student", hue="attack_name",
                                     label="Defended", ax=ax, palette="rainbow")
        
        # Customize legends
        handles, labels = ax.get_legend_handles_labels()
        #normal_legend = plt.legend(handles[:len(tmp_df['attack_name'].unique())], tmp_df['attack_name'].unique(),
        #                           title='Normal', fontsize=10, title_fontsize=12, bbox_to_anchor=(0.25, 1.02),
        #                           loc="upper left", ncol=1)
        undefended_legend = plt.legend(handles[2*len(tmp_df['attack_name'].unique()):3*len(tmp_df['attack_name'].unique())],
                                       tmp_df['attack_name'].unique(), title='Undefended', fontsize=10,shadow=True,fancybox=True,
                                       title_fontsize=12, bbox_to_anchor=(0.3, 1.02), loc="upper right", ncol=1)
        defended_legend = plt.legend(handles[4*len(tmp_df['attack_name'].unique()):],
                                     tmp_df['attack_name'].unique(), title='Defended', fontsize=10, title_fontsize=12,shadow=True,
                                     bbox_to_anchor=(0.55, 1.02), loc="upper right", ncol=1,fancybox=True)
    
        #ax.add_artist(normal_legend)
        ax.add_artist(undefended_legend)

        
        # Customize plot properties
        
        #plt.title(model_name)
        plt.ylim([-0.2,9])
        plt.grid()
        plt.xlabel(r'$\epsilon$-budget', fontsize=16)
        plt.ylabel(r'MSE', fontsize=16)
        #L = plt.legend(fontsize=12, shadow=True,ncols=3,
        #               bbox_to_anchor=(0.5, -0.92), loc="lower center") # using a size in points
        #for i in range(len(L.get_texts())):
        #    L.get_texts()[i].set_text(L.get_texts()[i].get_text().upper())
    
        #plt.setp(L.get_title(),fontsize='12')
        
        start_x = 0.0
        end_x = tmp_df.eps_val.values.max()
        x_axis_vals = np.linspace(start_x,end_x, tmp_df.mse_normal.size)
        plt.fill_between(x_axis_vals, 0, tmp_df.mse_normal.values, color='green', alpha=.2)
        plt.fill_between(x_axis_vals, 9.0, tmp_df.mse_normal.values, color='red', alpha=.2)
        plt.grid()
        
        plt.savefig(model_name + "-mitigation.pdf",bbox_inches='tight')
        
        
        plt.show()

    
if __name__ == "__main__":
    create_plots_for_mitigtaion('LSTM_model', 'bim',0.41)
    
    
    