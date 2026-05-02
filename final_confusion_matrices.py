import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'E:/projects/Anti-VEGF-0AD6')
from Model.CycleGan import Generator
from trainer.datasets import EyeDataset
from trainer.utils import ToTensor, Resize

def test_period(period, config):
    model = Generator(config['input_nc'], config['output_nc'])
    ckpt_path = f"checkpoint/Classifier_{period}_best_netG_A2B.pth"
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    val_transforms = [ToTensor(), Resize(size_tuple=(config['size'], config['size']))]
    dataset = EyeDataset(
        f'data/Classifier/{period}-term', 0,
        transforms_1=val_transforms, transforms_2=None,
        unaligned=False, type='test'
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    real_labels = []
    pred_labels = []
    test_input = torch.Tensor(16, config['input_nc'], config['size'], config['size'])
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Testing {period}"):
            batch_size = batch['A'].size(0)
            input_tensor = test_input[:batch_size]
            real_A = Variable(input_tensor.copy_(batch['A']))
            class_labels = batch['class_label'].cpu().numpy()
            
            _, pre_label = model(real_A)
            pred_classes = pre_label.detach().cpu().numpy().argmax(axis=1)
            
            real_labels.extend(class_labels.tolist())
            pred_labels.extend(pred_classes.tolist())
    
    return confusion_matrix(np.array(real_labels), np.array(pred_labels), labels=[0, 1, 2])

def plot_paper_style(all_cms, periods, save_path='output/all_confusion_matrices.png'):
    """绘制论文风格的并排混淆矩阵"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    class_names = ['Stable', 'Ineffective', 'Effective']
    
    for idx, (period, cm) in enumerate(zip(periods, all_cms)):
        ax = axes[idx]
        im = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
        
        # 设置刻度
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(class_names, rotation=45, ha='left')
        ax.set_yticklabels(class_names, rotation=45, va='center')
        
        # 添加数值标注
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]),
                       ha='center', va='center',
                       fontsize=14, fontweight='bold',
                       color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(f'{period}-term', fontsize=14, fontweight='bold')
        
        # 计算准确率
        acc = np.trace(cm) / np.sum(cm)
        ax.text(0.5, -0.2, f'Accuracy: {acc:.3f}',
               transform=ax.transAxes, ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"论文风格混淆矩阵已保存: {save_path}")
    plt.show()

if __name__ == '__main__':
    config = {'input_nc': 1, 'output_nc': 1, 'size': 256, 'num_classes': 3}
    
    periods = ['Short', 'Mid', 'Long']
    all_cms = []
    
    for period in periods:
        print(f"\n处理 {period}-term...")
        cm = test_period(period, config)
        all_cms.append(cm)
        print(f"混淆矩阵:\n{cm}")
        print(f"准确率: {np.trace(cm)/np.sum(cm):.3f}")
    
    plot_paper_style(all_cms, periods)