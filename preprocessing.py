import cv2
import numpy as np
from torchvision.transforms import functional, RandomCrop

def p_normalize(image):
    """
    对图像进行归一化处理
    
    参数：
    - image: 输入图像，灰度图像（单通道）或者彩色图像（三通道）
    - cv2.normalize:
        src: 输入图像。
        dst: 输出图像，如果设置为 None，则结果会被保存在原始图像中。
        alpha: 缩放因子的比例系数，默认为1。它用于控制归一化的放大倍数。
        beta: 偏移量的值，即结果中的像素值会加上这个值，默认为0。它用于控制归一化的偏移量。
        norm_type: 归一化的类型，指定归一化的方法。常用的类型有：
            cv2.NORM_MINMAX: 按照最小最大值进行归一化，将输入图像线性变换到指定的范围内。
            cv2.NORM_INF: 将输入图像的每个像素值除以其绝对值的最大值。
            cv2.NORM_L1: 将输入图像的每个像素值除以其绝对值的和。
            cv2.NORM_L2: 将输入图像的每个像素值除以其绝对值的平方和的平方根。
        dtype: 输出图像的数据类型，默认为 -1，表示与输入图像的数据类型保持一致。
    
    返回值：
    - normalized_image: 归一化后的图像
    """
    # 检查输入图像是否为空
    if image is None:
        print("Input image is None. Please provide a valid image.")
        return None
    
    # 对图像进行归一化处理
    normalized_image = cv2.normalize(image, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    return normalized_image

def p_filter(image, filter_type: str, kernel_size: int = 5) -> None:
    """
    对图像进行滤波处理，根据不同的滤波类型选择不同的滤波器
    :param filter_type: 滤波类型，类型为字符串，可选值为'median', 'mean', 'gaussian', 'bilateral', 'guided'等
    :param kernel_size: 滤波器的核大小，类型为整数，必须为奇数
    :return: None
    """
    if filter_type == 'median':
        # 使用中值滤波器，去除椒盐噪声
        image_blur = cv2.medianBlur(image, kernel_size)
    elif filter_type == 'mean':
        # 使用均值滤波器，平滑图像
        image_blur = cv2.blur(image, (kernel_size, kernel_size))
    elif filter_type == 'gaussian':
        # 使用高斯滤波器，平滑图像，保留更多细节
        image_blur = cv2.GaussianBlur(
            image, (kernel_size, kernel_size), 0)
    elif filter_type == 'bilateral':
        # 使用双边滤波器，平滑图像，保留边缘
        image_blur = cv2.bilateralFilter(
            image, kernel_size, kernel_size * 2, kernel_size / 2)
    elif filter_type == 'guided':
        # 使用导向滤波器，平滑图像，保留边缘和细节
        image_blur = cv2.ximgproc.guidedFilter(
            image, image, kernel_size, 0.1)
    else:
        # 无效的滤波类型，抛出异常
        raise ValueError("Invalid filter type: {}".format(filter_type))
    return image_blur


    
def p_grayscale_transform(image):
    # 灰度反转
    output_image = cv2.bitwise_not(image)
    return output_image

def p_log_transform(image, c=1.0):
    # 转换图像到浮点数类型，以避免溢出或截断错误
    float_image = image.astype(np.float32)
    
    # 应用对数变换公式
    log_image = c * np.log(1 + float_image)
    
    # 归一化结果，使其落在0-255范围内，并接收返回值
    normalized_image = cv2.normalize(log_image, None, 0, 255, cv2.NORM_MINMAX)
    
    # 转换回原始图像类型
    output_image = np.uint8(normalized_image)
    return output_image

def p_gamma_transform(image, gamma=1.0, c=1.0):
    # 转换图像到浮点数类型
    float_image = image.astype(np.float32)
    
    # 应用伽马变换公式
    gamma_image = c * np.power(float_image, gamma)
    # 归一化结果，使其落在0-255范围内，并接收返回值
    normalized_image = cv2.normalize(gamma_image, None, 0, 255, cv2.NORM_MINMAX)
    # 转换回原始图像类型
    output_image = np.uint8(normalized_image)
    return output_image

def p_histogram_equalization(image):
    """
    对输入图像进行直方图均衡化
    """
    # 读取图像
    # 应用直方图均衡化
    output_image = cv2.equalizeHist(image)
    return output_image

def p_adaptive_histogram_equalization(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    对输入图像进行自适应直方图均衡化
    参数：
    - clip_limit: 限制对比度的阈值，默认为2.0
    - tile_grid_size: 小块的大小，默认为(8, 8)
    返回值：
    - clahe_equalized_image: 自适应均衡化后的图像
    """


    # 创建自适应直方图均衡化对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # 应用自适应直方图均衡化
    output_image = clahe.apply(image)
    return output_image



def p_show_frequency_domain_image(image, image_shape):
    """
    显示图像的频域图像
    """

    img_float = np.float32(image)

    # 计算图像的离散傅里叶变换
    f = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)

    # 将零频分量移动到数组的中心
    fshift = np.fft.fftshift(f)

    # 计算频域图像的幅度谱
    magnitude = cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1])

    # 对幅度谱进行对数变换
    log_magnitude = np.log(magnitude + 1)

    # 对对数幅度谱进行归一化
    norm_magnitude = cv2.normalize(log_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # 创建一个图像网格
    # 设置低通滤波器
    rows, cols = image_shape
    # 中心位置
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
    
    # 掩膜图像和频谱图像乘积
    
    filtered_fshift = fshift * mask
    ishift = np.fft.ifftshift(filtered_fshift)
    iimg = cv2.idft(ishift)
    iimg_magnitude = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])

    # 计算低通滤波后的频域图像
    filtered_magnitude = cv2.magnitude(filtered_fshift[:, :, 0], filtered_fshift[:, :, 1])
    log_filtered_magnitude = np.log(filtered_magnitude + 1)
    norm_filtered_magnitude = cv2.normalize(log_filtered_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # 使用for循环构建plt图像
    images = [norm_magnitude, norm_filtered_magnitude, iimg_magnitude]
    titles = ['Frequency Domain Image', 'Low Pass Filtered Frequency Domain Image', 'Inverse Fourier Transform Image']
    return images, titles


def p_morphological_operation(image, operation_type, kernel_size=(5, 5)):
    """
    Performs morphological operations (erosion, dilation, opening, closing, top hat, black hat) 
    on the input image based on the specified operation type.

    Args:
        image: Input image (numpy array).
        operation_type: String specifying the type of morphological operation.
                        Supported types: 'erosion', 'dilation', 'opening', 'closing', 'top_hat', 'black_hat'
        kernel_size: Tuple specifying the size of the kernel (default is (5, 5)).

    Returns:
        Processed image based on the specified morphological operation.
    """
    kernel = np.ones(kernel_size, np.uint8)
    if operation_type == 'erosion':
        return cv2.erode(image, kernel, iterations=1)
    elif operation_type == 'dilation':
        return cv2.dilate(image, kernel, iterations=1)
    elif operation_type == 'opening':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation_type == 'closing':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    elif operation_type == 'top_hat':
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    elif operation_type == 'black_hat':
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    else:
        raise ValueError("Unsupported operation type. Supported types: 'erosion', 'dilation', 'opening', 'closing', 'top_hat', 'black_hat'")
    

def wiener_filter(image, image_shape, kernel=None, K=0.01):
    """
    对图像应用Wiener滤波。
    :param img: 输入图像
    :param kernel: 退化函数（卷积核），默认为5x5平均核
    :param K: 噪声功率与信号功率的比率
    :return: 复原的图像
    """
    if kernel is None:
        kernel = np.ones((5, 5)) / 25

    dummy = np.copy(image)
    if len(image_shape) == 2:  # 灰度图
        padded_kernel = np.pad(kernel, [(0, dummy.shape[0] - kernel.shape[0]), (0, dummy.shape[1] - kernel.shape[1])], 'constant')
    else:
        raise ValueError("Image must be 2D (grayscale).")
    
    kernel_ft = np.fft.fft2(padded_kernel)
    img_ft = np.fft.fft2(dummy)
    
    # 应用Wiener滤波
    kernel_ft_conj = np.conj(kernel_ft)
    numerator = kernel_ft_conj
    denominator = kernel_ft * kernel_ft_conj + K
    wiener_ft = img_ft * numerator / denominator
    
    # 将结果转换回空域
    wiener = np.fft.ifft2(wiener_ft).real

     # 规范化图像以适应uint8格式
    wiener = np.clip(wiener, 0, 255)  # 限制数值范围避免数据溢出
    wiener = wiener.astype(np.uint8)  # 转换为uint8类型
    
    return wiener


def constrained_least_squares(image, image_shape, gamma = 0.01):
    #有约束最小二乘方

    # 确保PSF的中心在图像中心
    psf_size = (5, 5)  # PSF大小
    psf = np.ones(psf_size) / 25  # 创建PSF

    psf_padded = np.zeros_like(image)  # 用零填充一个与图像相同大小的数组
    # 计算PSF应该插入的中心位置
    startx = psf_padded.shape[0]//2 - psf_size[0]//2
    starty = psf_padded.shape[1]//2 - psf_size[1]//2
    # 将PSF放置在psf_padded的中心
    psf_padded[startx:startx+psf_size[0], starty:starty+psf_size[1]] = psf


    # 计算PSF的傅里叶变换
    psf_fft = np.fft.fft2(psf, s=image_shape)

    # Laplacian operator for regularization
    laplacian = np.array([[0, -1, 0], 
                          [-1, 4, -1], 
                          [0, -1, 0]])
    laplacian_padded = np.zeros(image_shape)
    laplacian_padded[:3, :3] = laplacian
    laplacian_padded = np.roll(laplacian_padded, -1, axis=0)
    laplacian_padded = np.roll(laplacian_padded, -1, axis=1)
    
    # 计算拉普拉斯算子的傅里叶变换
    laplacian_fft = np.fft.fft2(laplacian_padded)
    
    # 计算复原滤波器
    filter_fft = np.conj(psf_fft) / (np.abs(psf_fft)**2 + gamma * np.abs(laplacian_fft)**2)
    
    # 应用复原滤波器
    result_fft = filter_fft * np.fft.fft2(image)
    
    # 逆傅里叶变换得到最终复原图像
    result_image = np.fft.ifft2(result_fft).real
    result_image = np.abs(result_image)

    result_image = np.clip(result_image, 0, 255)  # 限制值在0到255之间S
    result_image = result_image.astype(np.uint8)  # 转换为uint8

    return result_image


def image_resize(image, width, height):
    # 调整图片大小
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    return resized_image

def image_crop(image, mask, height ,width):
    rect = RandomCrop.get_params(image, (height, width))
    image = functional.crop(image, *rect)
    mask = functional.crop(mask, *rect)
    return image, mask

