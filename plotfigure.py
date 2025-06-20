import numpy as np
import scipy.io as sio
import spectral
from utils import applyPCA,padWithZeros
from net_hy import HybridSN
import torch
import matplotlib.pyplot as plt

patch_size = 25
pca_components = 30
num_class = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = HybridSN(num_classes=num_class, self_attention=True).to(device)
net.eval()
net_params = torch.load("./log/net_params.pkl")
net.load_state_dict(net_params)  # 加载模型可学习参数

# load the original image
X = sio.loadmat('./data/Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('./data/Indian_pines_gt.mat')['indian_pines_gt']
height = y.shape[0]
width = y.shape[1]

X = applyPCA(X, numComponents= pca_components)
X = padWithZeros(X, patch_size//2)

# 逐像素生成softmax结果
outputs = np.zeros([height*width,num_class])

k = 0
for i in range(height):
    for j in range(width):
        image_patch = X[i:i+patch_size, j:j+patch_size, :]
        image_patch = image_patch.reshape(1,image_patch.shape[0],image_patch.shape[1], image_patch.shape[2], 1)
        X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
        #将tensor类型转为numpy类型
        prediction = np.squeeze(net(X_test_image).detach().cpu().numpy())
        #按行存储
        outputs[k] = prediction
        k += 1
    if i % 20 == 0:
        print('... ... row ', i, ' handling ... ...')
#将结果进行reshape，并保存
outputs = outputs.reshape(height*width,num_class)
sio.savemat('results.mat', {'outputs': outputs})
