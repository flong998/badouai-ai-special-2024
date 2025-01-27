#环境：python=3.9 、 conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 cpuonly -c pytorch


import torch
import torchvision.transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw

# 加载预训练模型
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 让模型适应本地环境，判断本地是CPU还是GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)


# 加载图像并进行预处理
def preprocess_image(image):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    return transform(image).unsqueeze(0)


# 进行推理
def infer(image_path):
    # 使用Image打开图像，并转化为RGB
    image = Image.open(image_path).convert("RGB")
    # 预处理图像，添加batch维度，即n
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction


# 显示结果
def show_result(image, prediction):
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    # 画图
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            top_left = (box[0], box[1])
            bottom_right = (box[2], box[3])
            draw.rectangle([top_left, bottom_right], outline='red', width=2)
            draw.text((box[0], box[1] - 10), str(label), fill='yellow')
    image.show()


# 使用示例
image_path = 'street.jpg'
prediction = infer(image_path)
image = Image.open(image_path)
image = show_result(image, prediction)
