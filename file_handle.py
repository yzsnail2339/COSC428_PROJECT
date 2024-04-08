# -*- coding: utf-8 -*-
#APP的文件处理主要用于软件
import pandas as pd
import numpy as np
import os

import torch

class FileHandler:
    def __init__(self):
        pass

    def open_file_to_dataframe(self, file_name, file_extensions, filepath, sheetname = None):
        """读取文件"""
        self.data_frame = None
        if file_extensions == "csv":
            if ".csv" in file_name:
                file_name = file_name[:-4]
            try:
                self.data_frame = pd.read_csv(filepath + file_name +".csv",encoding='utf-8',dtype=object)
            except UnicodeDecodeError:
                self.data_frame = pd.read_csv(filepath + file_name +".csv",encoding='gbk',dtype=object)
      
        else:
            if ".xlsx" in file_name:
                file_name = file_name[:-5]
            try:        
                self.data_frame = pd.read_excel(filepath+file_name+".xlsx", sheet_name=sheetname)
            except UnicodeDecodeError:
                self.data_frame = pd.read_csv(filepath + file_name +".xlsx",encoding='gbk',dtype=object)
   
    
    def open_files_to_dataframe(self, filename, filepath, sqlite_class, primary_key_dict={}, add_index = False):
        """直接读取所有的sheet从xlsx文件"""
        excel_file = pd.ExcelFile(filepath + filename+".xlsx")
        sheet_names = excel_file.sheet_names
        #print(sheet_names)
        
        try:
            for sheet_name in sheet_names:
                table_name = "tbl_"+sheet_name
                
                if sqlite_class.db_has_table(table_name):
                    pass
                else:
                    self.data_frame = excel_file.parse(sheet_name)
                    #self.transfor_dataframe_format()
                    
                    self.get_dataframe_column_hander()
                    primary_list = []
                    if add_index:
                        self.add_index_column_to_dataframe()
                        primary_list.append("INDEXES")
                        
                    if sheet_name in primary_key_dict:
                        primary_list.append(primary_key_dict[sheet_name])
                                        
                    sqlite_class.db_create_table(table_name, self.data_frame_header, primary_list = primary_list)
                    self.add_to_db(table_name, "append")
            return True
        except Exception as e:
            return False
            
    def get_hander(self):
        self.data_frame_header = self.data_frame.columns.values
        
    def get_hander_list(self):
        self.header_list = self.data_frame_header.tolist()
    
    def get_dataframe_shape(self):
        self.data_frame_shape = self.data_frame.shape
                
               
    def transfor_dataframe_format(self, to_format: int = 0, index_lists: list = []):
        if to_format == 0:#转为string
            if len(index_lists) == 0:
                self.data_frame = self.data_frame.astype(str)
            else:
                picked_column_name_list = []
                for index in index_lists:
                    picked_column_name_list.append(self.data_frame_header[index])
                self.data_frame[picked_column_name_list] = self.data_frame[picked_column_name_list].astype(str)

        elif to_format == 1:
            if len(index_lists) == 0:#不能全部转为int
                
                #self.data_frame = self.data_frame.astype(int)
                pass
            else:
                picked_column_name_list = []
                for index in index_lists:
                    picked_column_name_list.append(self.data_frame_header[index])
                self.data_frame[picked_column_name_list] = self.data_frame[picked_column_name_list].astype(int)            
            
                
    def add_index_column_to_dataframe(self):
        """在第一行插入index (索引)列"""
        self.data_frame['INDEXES'] = np.arange(0, self.data_frame.shape[0])
        self.get_dataframe_column_hander()
        
    
    def drop_columns(self, save_header_name):
        """输入留下的名称"""
        self.data_frame = self.data_frame[save_header_name]#['工号','姓名']
        self.get_dataframe_column_hander()

        
    def rename_header(self, new_header_name):
        self.data_frame.columns = new_header_name
    
    def extend_column(self, extend_header):
        """拿nan扩展dataframe"""
        self.data_frame = self.data_frame.reindex(columns = extend_header)
        #print(self.data_frame)
    
    def dataframe_to_numpy(self, column_index=slice(1, None)):
        """
        转换DataFrame的指定列为NumPy数组。
        column_index可以是一个整数（选择单列），或者是一个切片对象（选择多列）。
        """
        if isinstance(column_index, int):
            # 如果是单个整数，只选择一列
            self.numpy_data = self.data_frame.iloc[:, column_index].values
        else:
            # 如果是切片对象，选择多列
            self.numpy_data = self.data_frame.iloc[:, column_index].to_numpy()

    
    def dataframe_to_list(self, column_index):
        """
        转换后为每一行的数据,这样可以直接按行遍历
        column_index : -1 全部转为 lists
        """
        if column_index > -1:
            self.nowlist = self.data_frame.iloc[:,column_index].values.tolist()
        else:
            self.nowlist = self.data_frame.values.tolist()
        #print(self.nowlist)
        
    def check_path_exist(self, path, create = False):
        is_exist = os.path.exists(path)
        if not is_exist:#不存在
            if create:
                os.makedirs(path)
                return True
            else:
                return False
        else:
            return True
        
    def mapping_column_hander(self,column_mapping):
        #column_mapping是一个字典, key是原标头, value是现标头
        if column_mapping != None:
            if "INDEXES" in self.data_frame_header:
                column_mapping["INDEXES"] = "INDEXES"
            self.data_frame = self.data_frame.rename(columns=column_mapping)
            self.get_dataframe_column_hander()
            
            
    def delect_file(self, file_path):
        os.remove(file_path)
    
    def get_file_size(self, file_path):
        return str(os.path.getsize(file_path))
    
    def prepare_pytorch_dataset(self):
        data_set = self.numpy_data.astype(float).astype(int)
        target = data_set.squeeze()
        target_tensor = torch.from_numpy(target)
        target_tensor = target_tensor.long()
        return self.data_frame.iloc[:, 0].values, target_tensor

    def remove_one_hot_columns(self, columns):
        # 删除 one-hot 编码的列
        # 参数:
        #   - columns: 包含 one-hot 编码列的列表

        # 将 DataFrame 转换为 numpy 数组
        data_as_array = self.data_frame[columns].values
        # 找到每行中 one-hot 编码值最大的索引
        indices = np.argmax(data_as_array, axis=1)
        self.data_frame['category_index'] = indices
        self.data_frame.drop(columns=columns, inplace=True)


 