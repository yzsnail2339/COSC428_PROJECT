from PIL import Image
import numpy as np
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../")


def analyze_mask_classes(mask_array):
    """
    分析遮罩图像中的唯一类别。

    参数:
    mask_path (str): 遮罩图像的路径。

    返回:
    tuple: 包含两个元素的元组，第一个是包含所有唯一类别的NumPy数组，第二个是类别的总数。
    """
    if isinstance(mask_array, np.ndarray):
        pass
    else:
        mask_array = np.array(mask_array)



    # 找出数组中所有唯一的像素值（每个值代表一个类别）
    unique_classes = np.unique(mask_array)

    # 计算类别的总数
    num_classes = len(unique_classes)

    return unique_classes, num_classes

def analyze_mask_classes_rgb(mask_array):
    """
    分析RGB遮罩图像中的唯一类别。

    参数:
    mask_path (str): 遮罩图像的路径。

    返回:
    tuple: 包含两个元素的元组，第一个是包含所有唯一颜色的NumPy数组，第二个是颜色（类别）的总数。
    """
    # 使用PIL读取遮罩图像（默认为RGB模式）
    # mask = Image.open(mask_path).convert('RGB')

    # 将遮罩图像转换为NumPy数组
    if isinstance(mask_array, np.ndarray):
        print("mask_array是一个numpy数组")
        print(mask_array.shape)
    else:
        print("mask_array不是一个numpy数组")
        mask_array = np.array(mask_array)


    # 重塑数组，使其成为一个二维数组，其中每行代表一个像素的RGB值
    pixels = mask_array.reshape(-1, mask_array.shape[2])

    # 使用NumPy的unique函数找出唯一的行（即唯一的RGB颜色值）
    # 这里的axis=0参数表示在行的方向上找唯一值
    unique_colors = np.unique(pixels, axis=0)

    # 计算唯一颜色的数量，即类别的总数
    num_classes = len(unique_colors)

    return unique_colors, num_classes


def image2label_dynamic_pil(image):

    # 转换图像数据为numpy数组
    image_np = np.array(image, dtype='int64')
    
    # 提取唯一颜色并创建颜色映射
    unique_colors = np.unique(image_np.reshape(-1, image_np.shape[2]), axis=0)
    color_map = {tuple(color): i for i, color in enumerate(unique_colors)}
    
    # 创建一个与图像同形状的空数组，用于存放映射后的类别标签
    label_image = np.zeros(image_np.shape[:2], dtype=int)
    
    # 遍历颜色映射，映射像素到类别
    for color, label in color_map.items():
        # 找到当前颜色的所有像素位置，并为它们分配类别标签
        is_color = np.all(image_np == color, axis=-1)
        label_image[is_color] = label
    
    return label_image, color_map

# def generate_new_color(existing_colors):
#     # 这是一个示例函数，用于生成新颜色
#     # 实际应用中应该根据需要来定义如何选择新颜色
#     # 这里简单地返回一个随机颜色
#     new_color = tuple(np.random.randint(0, 256, size=3))
#     while new_color in existing_colors:
#         new_color = tuple(np.random.randint(0, 256, size=3))
#     return new_color

# def image_color_mapping(image, colormap):
#     # 将图像转换为numpy数组
#     image_array = np.array(image)
#     # 获取图像的唯一颜色及其索引
#     unique_colors, indices = np.unique(image_array.reshape(-1, image_array.shape[2]), axis=0, return_inverse=True)
#     unique_colors = [tuple(color) for color in unique_colors]
    
#     # 更新colormap，为新颜色生成映射
#     for color in unique_colors:
#         if color not in colormap:
#             colormap[color] = generate_new_color(colormap.values())

#     # 使用colormap更新图像颜色
#     mapped_colors = np.array([colormap[color] for color in unique_colors])
#     mapped_image_array = mapped_colors[indices].reshape(image_array.shape)
    
#     return mapped_image_array, colormap

def generate_new_color(existing_colors):
    # 计算现有颜色列表中的最大值
    
    max_existing_color = max(existing_colors) if existing_colors else 0
    # print(max_existing_color)
    # 新的值等于现有颜色列表中的最大值加1
    new_color = max_existing_color + 1
    return new_color

def image_color_mapping(image_array, colormap, backgroud_color_map = []):
 
    # 确保image_array是单通道，如果不是，则转换为灰度（这一步取决于您的具体需求，如果已知肯定是单通道可以省略）
    if image_array.ndim == 3:
        image_array = np.mean(image_array, axis=-1).astype(image_array.dtype)  # RGB到灰度的简单转换

    unique_colors, indices = np.unique(image_array.reshape(-1), axis=0, return_inverse=True)
    unique_colors = [color for color in unique_colors]
    # print(unique_colors, indices.shape)
    
  
    updated_colormap = colormap.copy()#backgroud_color_map是多个背景颜色例如边界或者背景，这里映射到同一个颜色，但是不能设置成多个类
    for color in unique_colors:  # 更新colormap，为新颜色生成映射
        if color in backgroud_color_map:
            colormap[color] = 0
            updated_colormap[0] = 0
        else:
            if color not in colormap:
                # 确保colormap.values()可以正常迭代
                existing_colors = list(colormap.values())
                new_color = generate_new_color(existing_colors)
                colormap[color] = new_color
                updated_colormap[color] = new_color
    # 使用colormap更新图像颜色

    mapped_colors = np.array([colormap[color] for color in unique_colors])
    mapped_image_array = mapped_colors[indices].reshape(image_array.shape)  # 保持原有形状
    
    # 确保返回单通道图像
    # return colormap
    return mapped_image_array, updated_colormap

def display_image_with_pixel_values(mask_np_array, annotate_every=1, text_size=8):
    """
    Displays an image and annotates each pixel with its value, skipping pixels if needed.
    
    Parameters:
    - mask_np_array: A NumPy array containing the image data.
    - annotate_every: Annotate every nth pixel to avoid clutter (default 1, no skipping).
    - text_size: Size of the text annotation.
    """
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(mask_np_array, cmap='gray')

    # Determine the color for text annotation based on pixel brightness
    max_val = np.max(mask_np_array)
    
    # Annotate the pixels with their values
    rows, cols = mask_np_array.shape
    for y in range(0, rows, annotate_every):
        for x in range(0, cols, annotate_every):
            pixel_value = mask_np_array[y, x]
            ax.text(x, y, str(pixel_value), color="red", fontsize=text_size,
                    ha='center', va='center')
    
    plt.axis('off')
    plt.show()

# def generate_new_color(existing_colors):
#     # 这是一个示例函数，用于生成新颜色
#     # 实际应用中应该根据需要来定义如何选择新颜色
#     # 这里简单地返回一个随机颜色
#     new_color = tuple(np.random.randint(0, 256, size=3))
#     while new_color in existing_colors:
#         new_color = tuple(np.random.randint(0, 256, size=3))
#     return new_color

# def image_color_mapping(image, colormap):
#     # 将图像转换为numpy数组
#     image_array = np.array(image)
#     # 获取图像的唯一颜色及其索引
#     unique_colors, indices = np.unique(image_array.reshape(-1, image_array.shape[2]), axis=0, return_inverse=True)
#     unique_colors = [tuple(color) for color in unique_colors]
    
#     # 更新colormap，为新颜色生成映射
#     for color in unique_colors:
#         if color not in colormap:
#             colormap[color] = generate_new_color(colormap.values())

#     # # 使用colormap更新图像颜色
#     # mapped_colors = np.array([colormap[color] for color in unique_colors])
#     # mapped_image_array = mapped_colors[indices].reshape(image_array.shape)
#     return colormap
#     # return mapped_image_array, colormap

# if __name__  == "__main__":
#     from deep_learning.dp_support import generate_color_map
#     # num_classes = 20
#     # color_map = generate_color_map(num_classes)
#     # print(color_map)
#     mask_path = r'E:\UQ\Comp7840\dataset\gtFine_trainvaltest\gtFine\train\strasbourg\strasbourg_000001_034494_gtFine_labelTrainIds.png'  # 替换为你的遮罩图像路径
#     # mask_path = r'E:\UQ\Comp7840\dataset\VOCdevkit\VOC2012\SegmentationClass\2007_000032.png'
#     mask = Image.open(mask_path).convert('RGB')
#     mask, colormap = image_color_mapping(mask, {})
#     print(colormap)
#     plt.figure(figsize=(10, 5))

#     plt.subplot(1, 2, 1)
#     plt.title("Original Image")
#     plt.imshow(mask)
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.title("Mapped Image")
#     plt.imshow(mask)
#     plt.axis('off')

#     plt.show()
#     # channels = mask.mode

#     # # # 打印通道数
#     # print("图像通道数:", channels)
#     # unique_classes, raw_num_classes = analyze_mask_classes_rgb(mask)
#     # print(raw_num_classes)
#     # label_image, dynamic_colormap = image2label_dynamic_pil(mask)
#     # # label_image = image2label(label_image, dynamic_colormap)
#     # unique_classes = np.unique(label_image)
#     # image2label_num_classes = len(unique_classes)
#     # print(image2label_num_classes)
#     # # assert image2label_num_classes == raw_num_classes
