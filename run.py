import os
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
import numpy as np
from torch import nn
import torch
from unet import UNet
from loss import cross_entropy2d
from preprocessing import image_crop
from image_handle import analyze_mask_classes, analyze_mask_classes_rgb, image_color_mapping
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from dp_support import *
from path_handle import find_files_with_extension



def calculate_iou(preds, labels, num_classes):
    """
    计算给定预测和标签的IoU。
    
    参数:
    preds: 模型的预测输出，形状为[N, H, W],N:批量大小,H:图像的高度,W:图像的宽度。
    labels: 真实的标签,形状应与preds相同。
    num_classes: 数据集中的类别数。
    
    返回:
    一个字典,包含每个类别的IoU和平均IoU。
    """
    iou_per_class = []
    for cls in range(num_classes):
        preds_cls = preds == cls
        labels_cls = labels == cls
        
        intersection = (preds_cls & labels_cls).sum()
        union = (preds_cls | labels_cls).sum()
        
        if union == 0:
            # 避免除以0的错误
            iou = float('nan')
        else:
            iou = intersection / union
        iou_per_class.append(iou)
    
    # 计算所有类别的平均IoU，忽略nan值
    mean_iou = np.nanmean(iou_per_class)
    return {"iou_per_class": iou_per_class, "mean_iou": mean_iou}


# class CustomResizeTransform:
#     def __init__(self):

#         self.base_size = 224

#     def __call__(self, img):
#         w, h = img.size
#         # 判断图像的长宽比，并选择适当的尺寸进行缩放
#         if w / h > 1.5:  # 宽图像，长宽比接近2:1
#             size = (448, 224)
#         elif h / w > 1.5:  # 高图像，长宽比接近1:2
#             size = (224, 448)
#         else:  # 长宽比接近1:1
#             size = (224, 224)
#         return resize(img, size)

class UnetDataset(Dataset):
    #当你使用DataLoader时，它会自动为你的数据批量添加一个批次维度（batch dimension）?

    def __init__(self, image_path_list: list, label_path_list: list, file_list_length: int):
        self.transform = transforms.Compose([
            # CustomResizeTransform(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        # self.mask_transform = transforms.Compose([
        #     # CustomResizeTransform(),
        #     transforms.ToTensor(),
        # ])
        self.image_path_list = image_path_list
        self.label_path_list = label_path_list
        self.file_list_length = file_list_length
        self.color_map = {0: 0}
        self.width = self.height = 500


    def __getitem__(self, i: int):
        # print(self.label_path_list[i])
        image = Image.open(self.image_path_list[i]).convert('RGB')
        mask = Image.open(self.label_path_list[i]).convert('L')
        
        
        image_width, image_height = image.size
        if image_width < self.width: #300< 500
            if  image_height < self.height:
                image = image.resize((self.width, self.height), Image.BILINEAR)
                mask = mask.resize((self.width, self.height), Image.NEAREST)
            else:
                image = image.resize((self.width, image_height), Image.BILINEAR)
                mask = mask.resize((self.width, image_height), Image.NEAREST)
        else:
            if  image_height < self.height:
                image = image.resize((image_width, self.height), Image.BILINEAR)
                mask = mask.resize((image_width, self.height), Image.NEAREST)
            
        # image = image.resize((self.width, self.height), Image.NEAREST)
        # mask = mask.resize((self.width, self.height), Image.NEAREST)

        # print(np.array(image).shape)

        # print(analyze_mask_classes(mask))
        
        mask_np_array, self.color_map = image_color_mapping(np.array(mask), self.color_map)
        # for i in mask_np_array.tolist():
        #     print(i)
        # print(self.color_map)
        # expanded_mask = np.expand_dims(mask, axis=2)
        
      
        if self.transform != None: 

            image = self.transform(image)
            mask =  torch.from_numpy(mask_np_array)
            # mask = self.mask_transform(mask)
        # 将mask从[1, H, W]压缩为[H, W]，去掉单一的通道维度
        image, mask = image_crop(image ,mask ,height=self.height , width=self.width)
        # mask = torch.squeeze(mask, 0)
        mask = mask.squeeze(0).long()
        # for i in mask.tolist():
        #     print(i)
 
        # numpy_array = mask.numpy()

        # # 打印 NumPy 数组
        # print(numpy_array)
        # plt.imshow(mask, cmap='gray')  # 假设 mask 是灰度图像
        # plt.title('Mask')
        # plt.axis('off')
        # plt.show()

        return image, mask, self.color_map
        
    def __len__(self):
        return self.file_list_length



def decode_segmentation_mask(mask: np.ndarray, color_map: dict, name: str):
    """
    将颜色从现有的颜色映射回原始的颜色，并返回相应的颜色掩码
    :param mask: 2维的掩码
    :param color_map: 原颜色到现在的颜色的映射
    :param name: 分割掩码的名称
    :return: 对应的颜色掩码
    """
    # print(name, analyze_mask_classes(mask))
    replaced_image = np.copy(mask)
    for orig_color, new_color in color_map.items():
        replaced_image[mask == new_color[0]] = orig_color
    # print(analyze_mask_classes(replaced_image),mask.shape)
    return replaced_image


def save_prediction_vs_label(labels, preds, epoch: int, save_dir: str, color_map: dict):
    """
    保存预测掩码和真实掩码图像。
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    # print("colors :", color_map)
    
    for i in range(min(len(preds), 5)):  # 保存批次中前5个图像的预测和标签，以限制输出数量
        label_color_img = decode_segmentation_mask(labels[i].numpy(), color_map,"label")
        pred_color_img = decode_segmentation_mask(preds[i].numpy(), color_map, "pred")
        # print(preds[i].numpy())
        # print(preds[i].shape, pred_color_img.shape, label_color_img.shape)
        # print("结束\n")
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(label_color_img)
        ax[0].set_title('Ground Truth (labels)')
        ax[1].imshow(pred_color_img)
        ax[1].set_title('Prediction')

        
        # 保存图像
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_image_{i}.png"))
        plt.close()

# 




def train_model(model, device, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                optimizer, 
                num_epochs: int, start_epochs:int, num_classes: int, fcn_step: int, start_lr: float, task_name: str):
    best_loss = 1e10
    best_accuracy = 0.0  # 初始化最佳准确率
    best_iou = 0.0
    best_epoch = 0       # 初始化最佳epoch

    print("Task name:", task_name)
    # criterion = nn.CrossEntropyLoss()
    for epoch in range(start_epochs, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        # 每个epoch有两个训练阶段：训练和验证
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
                dataloader = train_loader
            else:
                model.eval()   # 设置模型为评估模式
                dataloader = val_loader
                preds_list = []
                labels_list = []


            running_loss = 0.0
            running_correct_pixels = 0
            total_pixels = 0

            # 迭代数据
            for i, data in enumerate(tqdm(dataloader, mininterval=5)):

                images, labels, color_map = data 
                images, labels = images.to(device), labels.to(device)
                # 前向传播
                # 跟踪历史仅在训练时
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    # loss = criterion(outputs, labels)
                    loss = cross_entropy2d(outputs, labels, size_average = True)
                    # 这里根据批次大小调整损失，使其成为平均每个样本的损失
                    # loss /= images.size(0)

                    # 后向传播 + 优化仅在训练阶段
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()



                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                running_correct_pixels += torch.sum(preds == labels).item()
                total_pixels += torch.numel(labels)
                if phase == 'val':
                    labels_list.append(labels)
                    preds_list.append(preds)

    

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_accuracy = (running_correct_pixels / total_pixels) * 100

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}, '
                f'Memory allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB, '
                f'Max memory allocated: {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB')

            print(f"{phase}_color_map:", color_map.keys())
            torch.cuda.reset_peak_memory_stats(device)  # 重置峰值内存统计，为下一个epoch准备

            if phase == 'val':

                # 阶段末尾，根据需要将所有Tensor转移到CPU
                all_labels = torch.cat(labels_list, dim=0).cpu()

                all_preds = torch.cat(preds_list, dim=0).cpu()

                variable_name = f"saved_predictions_{phase}"
                if fcn_step != 0:
                    variable_name += f"_{fcn_step}"

                save_prediction_vs_label(all_labels, all_preds, epoch, f"{task_name}_crop/project/{variable_name}", color_map)
                
                # 计算IoU，假设calculate_iou函数接受Tensor作为输入
                iou = calculate_iou(all_preds, all_labels, num_classes)
                print(f"Iou per Class: {iou['iou_per_class']}, Mean IoU: {iou['mean_iou']:.4f}")
                
                # 更新最佳模型（基于IoU）
                if iou['mean_iou'] > best_iou:
                    best_iou = iou['mean_iou']
                    best_loss = epoch_loss
                    best_epoch = epoch
                    save_model_state(model, optimizer=optimizer,epoch=epoch, save_path=f"{task_name}_crop/project/module/{best_epoch}.pth")
                    print(f"Saved new best model at epoch {best_epoch+1} with Mean IoU: {best_iou:.4f}")

        
            else:
                for param_group in optimizer.param_groups:
                    print(f"Learning Rate: {param_group['lr']}")
                    break
            

    print(f'Best epoch {best_epoch}, Best Loss {best_loss}')


if __name__ == '__main__':




    # train_image_dir = r".\image"
    # train_label_dir = r".\mask"
    train_image_dir = r"D:\UC_Master\COSC428\project\datas_first\training\images"
    train_label_dir = r"D:\UC_Master\COSC428\project\datas_first\training\v2.0\labels"

    
    train_image_file_name_list,train_image_file_name_list_length = find_files_with_extension(train_image_dir,[".jpg"])
    train_label_file_name_list, train_label_file_name_list_length = find_files_with_extension(train_label_dir, [".png"])
    
    

    # def read_image_path(root='E:/UQ\Comp7840/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'):
    #     """ Saving path under root """
    #     image = np.loadtxt(root, dtype=str)
    #     n = len(image)

    #     data_path, label_path = [], []
    #     counter = 0
    #     for i, name in enumerate(image):
    #         data_path.append(f'E:/UQ/Comp7840/dataset/VOCdevkit/VOC2012/JPEGImages/{name}.jpg')
    #         label_path.append(f'E:/UQ/Comp7840/dataset/VOCdevkit/VOC2012/SegmentationClass/{name}.png')
    #         if counter == 100:
    #             break
    #         else:
    #             counter += 1
    #     return data_path, label_path,counter
    # data_path, label_path, counter = read_image_path()
    # train_image_file_name_list,train_image_file_name_list_length = data_path, counter
    # train_label_file_name_list, train_label_file_name_list_length = label_path, counter
    # val_image_file_name_list, val_image_file_name_list_length = data_path, counter
    # val_label_file_name_list, val_label_file_name_list_length = label_path, counter


    torch.cuda.empty_cache()
    # 检查训练数据集图像和标签文件列表长度是否一致
    if train_image_file_name_list_length == train_label_file_name_list_length:
        # 继续执行
        pass
    else:
        # 退出并报错
        raise ValueError("Training image file list length does not match training label file list length.")

    # val_image_dir = r".\image"
    # val_label_dir = r".\mask"
    val_image_dir = r"D:\UC_Master\COSC428\project\datas_first\validation\images"
    val_label_dir = r"D:\UC_Master\COSC428\project\datas_first\validation\v2.0\labels"


    # 找到验证数据集图像和标签文件列表
    val_image_file_name_list, val_image_file_name_list_length = find_files_with_extension(val_image_dir, [".jpg"])
    val_label_file_name_list, val_label_file_name_list_length = find_files_with_extension(val_label_dir, [".png"])

    # # 检查验证数据集图像和标签文件列表长度是否一致
    if val_image_file_name_list_length == val_label_file_name_list_length:
        # 继续执行
        pass
    else:
        # 退出并报错
        raise ValueError("Validation image file list length does not match validation label file list length.")

    num_classes = 124
    # print(color_map)
    train_dataset = UnetDataset(
        train_image_file_name_list, 
        train_label_file_name_list,
        file_list_length = train_image_file_name_list_length,
    )
    val_dataset = UnetDataset(
        val_image_file_name_list, 
        val_label_file_name_list, 
        file_list_length = val_image_file_name_list_length,
    )
    batch_size = 2
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])
    train_loader = DataLoader(train_dataset, 
                                batch_size=batch_size,
                                num_workers= num_workers,
                                pin_memory=True
                              )
    
    val_loader = DataLoader(val_dataset, 
                                batch_size=batch_size,
                                num_workers= num_workers,
                                pin_memory=True
                            )


    device = pytorch_test()
    load_module = False
    if device != None:
        step = 0
        task_name = "unet"

        model = UNet(num_classes).to(device)


        lr, num_epochs = 0.01, 50
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if load_module:
            start_epochs = load_module_state(model, optimizer, "./unet_crop\project\module\0.pth")
        else:
            start_epochs = 0
        

        train_model(model, device, 
                    train_loader, val_loader,
                    optimizer,num_epochs, start_epochs= start_epochs, num_classes=num_classes, fcn_step = step, start_lr= lr, task_name = task_name)