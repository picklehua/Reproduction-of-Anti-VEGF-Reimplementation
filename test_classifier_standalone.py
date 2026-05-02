import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'E:/projects/Anti-VEGF-0AD6')
from Model.CycleGan import Generator
from trainer.datasets import EyeDataset
from trainer.utils import ToTensor, Resize

def test_period(period, config):
    print(f"\n{'='*60}")
    print(f"正在测试: {period}")
    print('='*60)

    model = Generator(config['input_nc'], config['output_nc'])
    ckpt_path = f"checkpoint/Classifier_{period}_best_netG_A2B.pth"
    state_dict = torch.load(ckpt_path, map_location='cpu')

    model.load_state_dict(state_dict)
    model.eval()

    print(f"模型加载成功，共 {len(state_dict)} 个层")

    val_transforms = [ToTensor(), Resize(size_tuple=(config['size'], config['size']))]
    dataset = EyeDataset(
        f'data/Classifier/{period}-term',
        0,
        transforms_1=val_transforms,
        transforms_2=None,
        unaligned=False,
        type='test'
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"数据加载成功: {len(dataset)} 个样本")

    real_labels = []
    pred_labels = []
    test_input = torch.Tensor(1, config['input_nc'], config['size'], config['size'])

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            real_A = Variable(test_input.copy_(batch['A']))
            class_label = batch['class_label'].item()

            _, pre_label = model(real_A)
            pre_label_np = pre_label.detach().cpu().numpy().squeeze()

            if pre_label_np.ndim != 1 or len(pre_label_np) != 3:
                continue

            pred_class = int(pre_label_np.argmax())
            real_labels.append(class_label)
            pred_labels.append(pred_class)

            if (i+1) % 200 == 0:
                print(f"  处理进度: {i+1}/{len(dataset)}")

    if len(real_labels) == 0:
        print("没有有效预测结果")
        return None

    real_labels = np.array(real_labels)
    pred_labels = np.array(pred_labels)

    print(f"\n有效样本数: {len(real_labels)}")
    print(f"真实标签分布: {np.bincount(real_labels, minlength=3)}")
    print(f"预测标签分布: {np.bincount(pred_labels, minlength=3)}")

    C = confusion_matrix(real_labels, pred_labels, labels=[0, 1, 2])
    print(f"混淆矩阵:\n{C}")

    class_names = ['Stable', 'Ineffective', 'Effective']
    plt.figure(figsize=(8, 6))
    plt.matshow(C, cmap=plt.cm.Blues, fignum=1)
    plt.xticks(range(3), class_names)
    plt.yticks(range(3), class_names, rotation=90, va='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    for i in range(3):
        for j in range(3):
            plt.annotate(str(C[i, j]), xy=(j, i),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=14, fontweight='bold')

    save_path = f'output/{period}-term/confusion_matrix.png'
    os.makedirs(f'output/{period}-term', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")

    print("\n--- 性能指标 ---")
    for i in range(3):
        tp = C[i, i]
        fn = np.sum(C[i, :]) - tp
        fp = np.sum(C[:, i]) - tp
        tn = np.sum(C) - tp - fn - fp
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"{class_names[i]}: Sensitivity={sen:.3f}, Specificity={spe:.3f}")

    acc = np.trace(C) / np.sum(C)
    print(f"Overall Accuracy: {acc:.3f}")

    return C

if __name__ == '__main__':
    config = {
        'input_nc': 1,
        'output_nc': 1,
        'size': 256,
        'num_classes': 3
    }

    results = {}
    for period in ['Short', 'Mid', 'Long']:
        try:
            cm = test_period(period, config)
            if cm is not None:
                results[period] = cm
        except Exception as e:
            print(f"{period} 测试失败: {e}")

    print("\n" + "="*60)
    print(f"成功测试: {list(results.keys())}")
    print("所有混淆矩阵已保存到 output/ 目录")
    print("="*60)