import xlrd
import os

def convert_to_xl(path):
    import pandas as pd
    import numpy as np
    # Reading the csv file
    # df_new = pd.read_csv('Names.csv')
    df_new = pd.read_csv(path+'.csv')
    # saving xlsx file
    GFG = pd.ExcelWriter(path+'.xls')
    df_new.to_excel(GFG, index = False)
    GFG.save()


current_path = os.getcwd()
action_grading_path = current_path + '/ActionGradingusingWATDXLS/'
for grading in ['1.6','3.3','5.0','6.6','8.3','10']:
    for experiment_number in range(1,11):
        path = action_grading_path+'S1_A'+str(grading)+'_'+str(experiment_number)
        print('path:'+path)
        convert_to_xl(path)
        print('grading '+str(grading)+' experiment_no '+str(experiment_number)+' converted')
        os.system('rm '+path+'.csv')
