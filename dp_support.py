import numpy as np
import torch
import os


def save_model_state(module, optimizer, epoch, save_path, **kwargs):
    """
    Saves the model state to a specified path.
    
    Parameters:
    - module: A state dictionary of the model to be saved.
    - optimizer: The optimizer's state dictionary.
    - epoch: Current epoch number.
    - save_path: The path where the model state should be saved.
    - kwargs: Additional information to be saved with the model state.
    """

    dir_path = os.path.dirname(save_path)

    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)
    save_dict = {
        'model_state_dict': module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 
        'epoch': epoch
    }
    
    # Optionally save additional information
    save_dict.update(kwargs)
    
    torch.save(save_dict, save_path)
    # print(f"Model saved to {save_path}")
    

def load_module_state(model, optimizer, save_path):
    """
    Loads the model and optimizer states from a specified path.

    Parameters:
    - model: The model to load the state into.
    - optimizer: The optimizer to load the state into.
    - save_path: The path from where the model state should be loaded.

    Returns:
    - epoch: The epoch number at which the model was saved.
    """
    saved_model = torch.load(save_path)
    model.load_state_dict(saved_model['model_state_dict'])
    optimizer.load_state_dict(saved_model['optimizer_state_dict'])
    epoch = saved_model['epoch']
    return epoch


def pytorch_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print('Using version:', torch.__version__)
    print()
    
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        properties = torch.cuda.get_device_properties(device)
    
        print("Total memory:", properties.total_memory / (1024**3), "GB")
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        return device

    else:
        assert "pytorch没有使用gpu"
        return None

def generate_color_map(num_classes) -> np.ndarray:
    """
    生成颜色映射表。

    参数:
    - num_classes: 类别的数量。

    返回:
    - 一个形状为(num_classes, 3)的numpy数组，表示每个类别的颜色（RGB）。
    """
    np.random.seed(62)  # 设置随机种子以确保颜色的一致性
    color_map = np.random.randint(0, 256, (num_classes, 3), dtype=np.uint8)
    return color_map.tolist()



def convert_image_to_labels(image_array: np.ndarray, color_map: list) -> np.ndarray:
    """
    Converts an image to a label map based on a given color map.
    
    Args:
    - image: An Image object of shape (height, width, 3) with RGB values.
    - color_map: A list of RGB colors that map to specific labels.

    Returns:
    - A 2D array of shape (height, width) with label values.
    """
    # Create a temporary array to hold the mapping from color values to labels.
    color_to_label_map = np.zeros(256 ** 3, dtype=np.int64)
    
    # Populate the mapping array with label indices based on the color map.
    for label_index, color in enumerate(color_map):
        color_key = ((color[0] * 256 + color[1]) * 256 + color[2])
        color_to_label_map[color_key] = label_index
    
    # Convert the input image to an array of indices based on its RGB values.
    image_as_indices = ((image_array[:, :, 0] * 256 + image_array[:, :, 1]) * 256 + image_array[:, :, 2])
    print(image_as_indices.tolist())
    # Use the indices to map each pixel to its corresponding label.
    label_image = color_to_label_map[image_as_indices]
    return label_image


def label2image(label, color_map: np.ndarray):
    """ about label-img(0-20) and convert it to class-img (128, 0, 0)"""
    h, w = label.shape
    label = label.reshape(h*w, -1)
    image = np.zeros((h*w, 3), dtype='int32')
    for ii in range(len(color_map)):
        index = np.where(label == ii)
        image[index, :] = color_map[ii]
    return image.reshape(h, w, 3)


def inv_normalize_image(data):
    """ inverse the image normalizing """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    data = data.astype('float32') * std + mean
    # 限制data数据的 min=0, max=1
    return data.clip(0, 1)
