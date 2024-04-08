#这是图片处理成深度学习需要的一些函数
import os
import cv2
import numpy as np
import json
import path_handle
class convertImageToMask():
    def __init__(self) -> None:
        self.classes_color = {} #这里保存{类：颜色}
        self.used_colors = set()  # 已使用的颜色集合
        self.next_gray_value = 0
    
    def set_label_path(self, label_path):
        self.label_path = label_path
        self.output_path = label_path
    
    def get_label_path(self):
        return self.label_path
    
    def set_output_path(self, output_path):
        self.output_path = output_path
        
    
    def get_output_path(self):
        return self.output_path
         

    def __get_color_for_class(self, class_name):
        if class_name not in self.classes_color:
            # 如果类名不在字典中，则生成新颜色并添加到字典中
            new_color = self.__generate_new_color()
            self.classes_color[class_name] = new_color
            self.used_colors.add(new_color)
            return new_color
        return self.classes_color[class_name]
    
    def __generate_new_color(self):
        # 生成一个新颜色，确保它与已使用的颜色不重复
        while True:
            color = tuple(np.random.randint(0, 256, size=3).tolist())
            if color not in self.used_colors:
                return color
    
    def __get_gray_value(self, class_name):
        if class_name not in self.classes_color:
            # 分配一个新的灰度值给这个类别
            self.classes_color[class_name] = self.next_gray_value
            self.next_gray_value += 1
            # 确保灰度值不超过255
            if self.next_gray_value > 255:
                raise ValueError("超出了可分配的灰度值范围")
        return self.classes_color[class_name]

   
    def convert_label_to_mask(self, directory: str, json_file_name: str):

        json_file_path = os.path.join(directory, json_file_name)

        with open(json_file_path, "r", encoding='utf-8') as jsonf:
            jsonData = json.load(jsonf)
            img_h = jsonData["imageHeight"]
            img_w = jsonData["imageWidth"]
            
            # mask_line = np.zeros((img_h, img_w, 3), np.uint8)  # 假定一个3通道图像用于彩色掩码

            mask = np.zeros((img_h, img_w), dtype=np.uint8)  # 使用单通道图像
            gray_value = self.__get_gray_value("background")
            # mask = np.full((img_h, img_w), 255, dtype=np.uint8)
            for obj in jsonData["shapes"]:
                label = obj["label"]

                # color = self.__get_color_for_class(label)  # 获取或生成颜色
                # polygonPoints = np.array(obj["points"], np.int32)
                # cv2.drawContours(mask_line, [polygonPoints], -1, color, 3)

                gray_value = self.__get_gray_value(label)
                polygonPoints = np.array(obj["points"], np.int32)
                cv2.fillPoly(mask, [polygonPoints], color=gray_value)

            mask_file_name = os.path.splitext(json_file_name)[0] + "_mask.png"
            mask_file_path = os.path.join(directory, mask_file_name)
            cv2.imwrite(mask_file_path, mask)
            return mask_file_path

        
    def convert_labels_to_mask(self):
        self.directories = path_handle.list_directories(self.label_path)
        image_txt_path = os.path.join(self.label_path, 'image.txt')
        label_txt_path = os.path.join(self.label_path, 'label.txt')

        with open(image_txt_path, 'w') as image_file, open(label_txt_path, 'w') as label_file:
            for directory in self.directories:
                directory = os.path.join(self.label_path, directory)
                json_file_name_list = path_handle.list_files_with_paths(directory, ["json"], "end")
                jpg_file_name_list = path_handle.list_files_with_paths(directory, ["jpg","png"], "end")
                for json_file_name in json_file_name_list:
                    json_file_base_name = os.path.splitext(json_file_name)[0]
                    corresponding_jpg_file = json_file_base_name + ".jpg"

                    if corresponding_jpg_file in jpg_file_name_list:
                        mask_file_path = self.convert_label_to_mask(directory, json_file_name)
                        image_file.write(os.path.join(directory, corresponding_jpg_file) + '\n')
                        label_file.write(os.path.join(directory, mask_file_path) + '\n')
                        jpg_file_name_list.remove(corresponding_jpg_file)
        image_file.close()
        label_file.close()
        self.__output_config_file() 

    def __output_config_file(self):
        config_data = {
            "number_of_classes": len(self.classes_color),  # 类的数量
            "class_names": list(self.classes_color.keys()),  # 类的名称
            "class_colors": self.classes_color  # 类对应的颜色或灰度值
        }

        config_path = os.path.join(self.output_path, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as config_file:
            json.dump(config_data, config_file, ensure_ascii=False, indent=4)

        print(f"配置文件已生成在：{config_path}")

    def display_mask(self, mask):
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    jsonfileFolder = r"E:\UQ\Comp7840\dataset\project2024\images"

    convertImageToMask_class = convertImageToMask()
    convertImageToMask_class.set_label_path(jsonfileFolder)
    convertImageToMask_class.convert_labels_to_mask()


