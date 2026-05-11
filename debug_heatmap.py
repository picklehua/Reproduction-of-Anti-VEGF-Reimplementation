import sys; sys.path.insert(0, 'E:/projects/Anti-VEGF-0AD6')
import yaml, torch, cv2, numpy as np
from trainer.CycTrainer import Cyc_Trainer, returnCAM
import torchvision.transforms as transforms
from trainer.utils import ToTensor, Resize
from torch.autograd import Variable
import torch.nn as nn

print("1/5 加载配置...")
with open('Yaml/CycleGan.yaml') as f:
    config = yaml.safe_load(f)
config['val_dataroot'] = 'data/Classifier/Short-term'
config['save_root'] = 'output/Short-term/'

print("2/5 加载模型...")
trainer = Cyc_Trainer(config)
trainer.netG_A2B.load_state_dict(torch.load('checkpoint/Classifier_Short_best_netG_A2B.pth', map_location='cpu'))
trainer.netG_A2B.eval()

print("3/5 提取特征层和权重...")
model_features = nn.Sequential(*list(trainer.netG_A2B.children())[:-3], list(trainer.netG_A2B.children())[-2])
fc1 = trainer.netG_A2B.state_dict()['classifier_tail.0.weight'].cpu().numpy()
fc2 = trainer.netG_A2B.state_dict()['classifier_tail.2.weight'].cpu().numpy()
fc3 = trainer.netG_A2B.state_dict()['classifier_tail.4.weight'].cpu().numpy()

print("4/5 推理并计算 CAM...")
T = transforms.Compose([ToTensor(), Resize(size_tuple=(256, 256))])
img = cv2.imread('data/Classifier/Short-term/before/3896.jpg', 0)
trainer.test_input_A.zero_()
real_A = Variable(trainer.test_input_A.copy_(T(img).unsqueeze(0)))

with torch.no_grad():
    _, pre_label = trainer.netG_A2B(real_A)
    pred_class = pre_label.argmax().item()
    features = model_features(real_A).detach().cpu().numpy()
    CAMs = returnCAM(features, fc1, fc2, fc3)

print(f'   预测类别: {pred_class}')
print(f'   CAM 范围: {CAMs[pred_class].min():.0f} ~ {CAMs[pred_class].max():.0f}')

print("5/5 生成热图...")
img_disp = real_A.detach().cpu().squeeze().numpy()
img_disp = cv2.normalize(img_disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
h, w = img_disp.shape
img_disp_color = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)

CAMs[pred_class][CAMs[pred_class] < 150] = 0
heatmap = cv2.applyColorMap(cv2.resize(CAMs[pred_class], (w, h)), cv2.COLORMAP_JET)
result = heatmap * 0.5 + img_disp_color * 0.5
result_noflip = result.copy()
result = cv2.flip(cv2.transpose(result), 1)

cv2.imwrite('output/Short-term/heatmap_debug_flipped.jpg', result)
cv2.imwrite('output/Short-term/heatmap_debug_noflip.jpg', result_noflip)
print('完成！对比 heatmap_debug_flipped.jpg 和 heatmap_debug_noflip.jpg')
