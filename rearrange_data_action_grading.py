import xlrd
import os

def write_data_to_file(data, path):
    # print(data)
    f = open(path, "a")
    f.write(data)
    f.close()

def convert_to_xl(path):
    import pandas as pd
    import numpy as np
    # Reading the csv file
    df_new = pd.read_csv('Names.csv')
    # saving xlsx file
    GFG = pd.ExcelWriter('Names.xlsx')
    df_new.to_excel(GFG, index = False)
    GFG.save()


current_path = os.getcwd()
action_grading_path = current_path + '/ActionGradingusingWATDXLS/'
path_for_text_files = current_path + '/ReArrangedTextFiles/'
path_for_sensor_1_x = path_for_text_files + 'sensor_1_x.txt'
path_for_sensor_1_y = path_for_text_files + 'sensor_1_y.txt'
path_for_sensor_1_z = path_for_text_files + 'sensor_1_z.txt'
path_for_sensor_2_x = path_for_text_files + 'sensor_2_x.txt'
path_for_sensor_2_y = path_for_text_files + 'sensor_2_y.txt'
path_for_sensor_2_z = path_for_text_files + 'sensor_2_z.txt'
path_for_sensor_3_x = path_for_text_files + 'sensor_3_x.txt'
path_for_sensor_3_y = path_for_text_files + 'sensor_3_y.txt'
path_for_sensor_3_z = path_for_text_files + 'sensor_3_z.txt'
path_for_output = path_for_text_files + 'output.txt'
for grading in ['1.6','3.3','5.0','6.6','8.3','10']:
    for experiment_number in range(1,11):
        print('grading '+str(grading)+' experiment_no '+str(experiment_number)+' enetered')
        path = action_grading_path+'S1_A'+str(grading)+'_'+str(experiment_number)+'.xls'
        print('path:'+path)
        wb = xlrd.open_workbook(path)
        sheet = wb.sheet_by_index(0)
        for i in range(1,1001):
            time_stamp = 0
            ax_column = 1
            ay_column = 2
            az_column = 3
            gx_column = 4
            gy_column = 5
            gz_column = 6
            mx_column = 7
            my_column = 8
            mz_column = 9
            write_data_to_file(str(sheet.cell_value(i,ax_column)), path_for_sensor_1_x)
            write_data_to_file(' ', path_for_sensor_1_x)
            write_data_to_file(str(sheet.cell_value(i,ay_column)), path_for_sensor_1_y)
            write_data_to_file(' ', path_for_sensor_1_y)
            write_data_to_file(str(sheet.cell_value(i,az_column)), path_for_sensor_1_z)
            write_data_to_file(' ', path_for_sensor_1_z)
            write_data_to_file(str(sheet.cell_value(i,gx_column)), path_for_sensor_2_x)
            write_data_to_file(' ', path_for_sensor_2_x)
            write_data_to_file(str(sheet.cell_value(i,gy_column)), path_for_sensor_2_y)
            write_data_to_file(' ', path_for_sensor_2_y)
            write_data_to_file(str(sheet.cell_value(i,gz_column)), path_for_sensor_2_z)
            write_data_to_file(' ', path_for_sensor_2_z)
            write_data_to_file(str(sheet.cell_value(i,mx_column)), path_for_sensor_3_x)
            write_data_to_file(' ', path_for_sensor_3_x)
            write_data_to_file(str(sheet.cell_value(i,my_column)), path_for_sensor_3_y)
            write_data_to_file(' ', path_for_sensor_3_y)
            write_data_to_file(str(sheet.cell_value(i,mz_column)), path_for_sensor_3_z)
            write_data_to_file(' ', path_for_sensor_3_z)
        write_data_to_file('\n', path_for_sensor_1_x)
        write_data_to_file('\n', path_for_sensor_1_y)
        write_data_to_file('\n', path_for_sensor_1_z)
        write_data_to_file('\n', path_for_sensor_2_x)
        write_data_to_file('\n', path_for_sensor_2_y)
        write_data_to_file('\n', path_for_sensor_2_z)
        write_data_to_file('\n', path_for_sensor_3_x)
        write_data_to_file('\n', path_for_sensor_3_y)
        write_data_to_file('\n', path_for_sensor_3_z)
        switch = {
        '1.6':'1',
        '3.3':'2',
        '5.0':'3',
        '6.6':'4',
        '8.3':'5',
        '10':'6'
        }
        output_data = switch[str(grading)]+'\n'
        write_data_to_file(output_data, path_for_output)
