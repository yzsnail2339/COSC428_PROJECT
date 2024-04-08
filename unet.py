import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from preprocessing import image_crop
from image_handle import analyze_mask_classes, display_image_with_pixel_values, image_color_mapping

class Encoder(nn.Module):
    def __init__(self, num_channels, num_filters):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x_before_pool = x
        x = self.relu(self.bn2(self.conv2(x)))
        x_pooled = self.pool(x)
        return x_before_pool, x_pooled

class Decoder(nn.Module):
    def __init__(self, num_channels_up, num_channels_skip, num_filters, upsampling = "convtrans"):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(num_channels_up, num_filters, kernel_size=2, stride=2)

        # self.conv_after_up = nn.Conv2d(num_channels_up, num_filters, kernel_size=3, stride=1, padding=1)#这里是错的

        self.conv1 = nn.Conv2d(num_channels_skip + num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu2 = nn.ReLU(inplace=True)
        self.upsampling = upsampling

    def forward(self, x, x_skip): # x是上一大层来的输出, x_skip是左侧的输出
        # print(x.shape)
        if self.upsampling == "convtrans": #TransposeConv 
            x = self.up(x)
            # print(x.shape)
           
        elif self.upsampling == "interpolation": #Upsample+Conv2d ，这里是错的，upsampling没完成
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            # x = self.conv_after_up(x)
            # print(x.shape)
        else:
            raise ValueError("inValidation upsampling in Unet Decoder")
        

        # 计算跳跃连接的特征图与上采样后特征图的尺寸差异。
        diffY = x_skip.size()[2] - x.size()[2]
        diffX = x_skip.size()[3] - x.size()[3]
        # print(x_skip.size(), x.size())
        
        # 对上采样后的特征图进行填充，使其尺寸与跳跃连接的特征图匹配。
        # 这对于后续的连接操作是必要的。
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # print("x:",x.shape)

        x = torch.cat([x_skip, x], dim=1)
        # print("x2:",x.shape)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.down1 = Encoder(3, 64)
        self.down2 = Encoder(64, 128)
        self.down3 = Encoder(128, 256)
        self.down4 = Encoder(256, 512)
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024,  kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.upsampling = "convtrans"
        self.up1 = Decoder(1024, 512, 512, self.upsampling)
        self.up2 = Decoder(512, 256, 256, self.upsampling)
        self.up3 = Decoder(256, 128, 128, self.upsampling)
        self.up4 = Decoder(128, 64, 64, self.upsampling)
        self.last_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        outputs = {}
        x1, x = self.down1(x)
        # outputs["down1"] = x1
        x2, x = self.down2(x)
        # outputs["down2"] = x2
        x3, x = self.down3(x)
        # outputs["down3"] = x3
        x4, x = self.down4(x)
        # outputs["down4"] = x4
        x = self.center(x)
        # outputs["center"] = x
        x = self.up1(x, x4)
        # outputs["up1"] = x
        x = self.up2(x, x3)
        # outputs["up2"] = x
        x = self.up3(x, x2)
        # outputs["up3"] = x
        x = self.up4(x, x1)
        # outputs["up4"] = x
        x = self.last_conv(x)
        # outputs["final"] = x
        return x
    
def visualize_all_feature_maps(outputs, name_lists: list):
    for layer_name, tensor in outputs.items():
        if layer_name in name_lists:
            # tensor的形状是 [Batch Size, Channels, Height, Width]
            batch_size, num_channels, height, width = tensor.shape
            # 设置合理的子图大小
            num_rows = int(np.sqrt(num_channels))
            num_cols = int(np.ceil(num_channels / num_rows))
            print(num_rows, num_cols)

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
            fig.suptitle(f"{layer_name} Feature Maps")

            for i, ax in enumerate(axes.flat):
                if i < num_channels:
                    # 提取第i个通道的特征图并移至CPU
                    feature_map = tensor[0, i].cpu().detach().numpy()
                    ax.imshow(feature_map, cmap='gray')
                    ax.axis('off')
                else:
                    ax.axis('off')
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()


def visualize_image_and_prediction(original_img, prediction):
    """
    Visualizes an original image and its prediction side by side.
    
    Parameters:
    - original_img: The original image tensor of shape (C, H, W).
    - prediction: The prediction tensor of shape (H, W), containing class indices for each pixel.
    """
    # Normalize and prepare the original image
    original_img_np = original_img.cpu().numpy()
    img_show = np.transpose(original_img_np, (1, 2, 0))
    img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())

    # Prepare the prediction
    pred_image = prediction.cpu().numpy()
    # pred_image = display_image_with_pixel_values(pred_image)
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Display the original image
    axes[0].imshow(img_show)
    axes[0].set_title("Original Image")
    axes[0].axis('off')  # Hide axis ticks
    
    # Display the prediction
    axes[1].imshow(pred_image) 
    axes[1].set_title("Prediction")
    axes[1].axis('off')  # Hide axis ticks

    plt.show()


def load_image(image_path, mask_path, height, width):
    """
    Load an RGB image from the filesystem and convert it to a tensor suitable for model input.
    """
    # Define the image transformations: resize, convert to tensor, and normalize
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert("L")

    image_width, image_height = image.size


    if image_width < width:
        if  image_height < height:
            image = image.resize((width,height), Image.BILINEAR)
            mask = mask.resize((width,height), Image.NEAREST)
        else:
            image = image.resize((width,image_height), Image.BILINEAR)
            mask = mask.resize((width,image_height), Image.NEAREST)
    else:
        if  image_height < height:
            image = image.resize((image_width,height), Image.BILINEAR)
            mask = mask.resize((image_width,height), Image.NEAREST)
        


    mask, color_map = image_color_mapping(np.array(mask), {}, [0,220])
    # print(display_image_with_pixel_values(mask_np_array), color_map)
    # print(mask_np_array.shape, np.max(mask_np_array), color_map)
    # plt.imshow(mask_np_array, cmap='gray')  # 确保使用灰度色图
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()

    transform = transforms.Compose([
        # CustomResizeTransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    mask = mask.astype(np.float32)
    mask = torch.from_numpy(mask)
    image, mask = image_crop(image ,mask , height, width)

    # Convert to PyTorch tensor

 
    return image, mask.squeeze(0)

# def gray_grid():
#     gray_image = np.linspace(0, 255, 256).reshape(16, 16).astype(np.uint8)

#     # 使用 matplotlib 显示这个灰度图
#     plt.figure(figsize=(8, 8))
#     plt.imshow(gray_image, cmap='gray', interpolation='nearest')
#     plt.colorbar()  # 显示颜色条
#     plt.title("Linear Grayscale Image")
#     plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gray_grid()
    if device != None:
        image_path = r"E:\UQ\Comp7840\dataset\VOCdevkit\VOC2012\JPEGImages\2007_000032.jpg"
        mask_path = r"E:\UQ\Comp7840\dataset\VOCdevkit\VOC2012\SegmentationClass\2007_000032.png"
        # image_size = 64
        images,mask = load_image(image_path, mask_path, 360,480)
        images, mask = images.unsqueeze(0).to(device), mask.to(device)
 
        
        net = UNet(num_classes=20).to(device)  # Instantiate the model
        net.eval()

        # try:
        x, output = net(images) # Assuming 'final' is your last layer's logits
        _, preds = torch.max(x, 1)  # Get the predicted class indices for each pixel
        # visualize_all_feature_maps(output, ["final"])
        # visualize_image_and_prediction(images[0], preds[0])

        # except Exception as e:
        #     print(f"Error during model forward pass: {e}")