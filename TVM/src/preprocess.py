#!python ./preprocess.py
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np

img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

# 重设大小为 224x224
resized_image = Image.open(img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")

# ONNX 需要 NCHW 输入, 因此对数组进行转换
img_data = np.transpose(img_data, (2, 0, 1))

# 根据 ImageNet 进行标准化
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_stddev = np.array([0.229, 0.224, 0.225])
norm_img_data = np.zeros(img_data.shape).astype("float32")
for i in range(img_data.shape[0]):
      norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - imagenet_mean[i]) / imagenet_stddev[i]

# 添加 batch 维度
img_data = np.expand_dims(norm_img_data, axis=0)

# 保存为 .npz（输出 imagenet_cat.npz）
np.savez("../data/imagenet_cat", data=img_data)