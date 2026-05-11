import sys, os, yaml, torch, numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
sys.path.insert(0, 'E:/projects/Anti-VEGF-0AD6')
from trainer.CycTrainer import Cyc_Trainer1

with open('Yaml/CycleGan.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['dataroot'] = 'data/Regression/Short-term'
cfg['val_dataroot'] = 'data/Regression/Short-term'
cfg['save_root'] = 'output/Regression/'
cfg['batchSize'] = 1; cfg['n_cpu'] = 0
os.makedirs('output/Regression', exist_ok=True)

tr = Cyc_Trainer1(cfg)
tr.netG_A2B.load_state_dict(torch.load('checkpoint/Regression_Short_best_netG_A2B.pth', map_location='cpu'))

# 收集数据
data = []
with torch.no_grad():
    for i, batch in enumerate(tr.test_data):
        real_A = torch.autograd.Variable(tr.test_input_A.copy_(batch['A']))
        class_label = batch['class_label']
        pre_label = tr.netG_A2B(real_A)
        data.append([class_label.item(), pre_label.item()])
data = np.array(data)
y_true = data[:, 0]; y_pred = data[:, 1]

# 计算指标
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

# 拟合回归线
lr = LinearRegression()
lr.fit(y_true.reshape(-1,1), y_pred)
x_line = np.linspace(y_true.min(), y_true.max(), 100).reshape(-1,1)
y_line = lr.predict(x_line)

# 95% CI (简化版：残差标准差)
residuals = y_pred - y_true
std_res = np.std(residuals)

# 画图
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_true, y_pred, alpha=0.3, s=5, color='#2E86AB')
ax.plot([0, 0.6], [0, 0.6], 'k--', lw=1.5, label='1:1 Line')
ax.plot(x_line, y_line, 'red', lw=2, label=f'Line of best fit')
ax.fill_between(x_line.flatten(), y_line.flatten()-1.96*std_res, y_line.flatten()+1.96*std_res,
                alpha=0.15, color='red', label='95% CI')
ax.set_xlabel('True BCVA', fontsize=12)
ax.set_ylabel('Predicted BCVA', fontsize=12)
ax.set_title(f'BCVA Prediction (R²={r2:.4f}, MAE={mae:.4f})', fontsize=14)
ax.legend(fontsize=10)
ax.set_xlim([0, 0.65]); ax.set_ylim([0, 0.65])
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/Regression/BCVA_scatter.png', dpi=300)
plt.close()
print(f'BCVA散点图已保存: output/Regression/BCVA_scatter.png')
print(f'R²={r2:.4f}, MAE={mae:.4f}')
