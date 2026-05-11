import sys, os, yaml, torch, cv2, numpy as np
from tqdm import tqdm
sys.path.insert(0, 'E:/projects/Anti-VEGF-0AD6')
from Model.CycleGan import Generator
from trainer.datasets import EyeDataset
from trainer.utils import ToTensor, Resize
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.utils import resample
import torchvision.transforms as transforms
from trainer.CycTrainer import Cyc_Trainer, Cyc_Trainer1, returnCAM
import shap

config = {'input_nc': 1, 'output_nc': 1, 'size': 256, 'num_classes': 3}
class_names = ['Stable', 'Ineffective', 'Effective']

def run_classification(period):
    """ROC, PR, Boxplot, Youden"""
    print(f"\n{'='*60}\n[Classification] {period}-term\n{'='*60}")
    os.makedirs(f'output/{period}-term', exist_ok=True)

    model = Generator(1, 1)
    model.load_state_dict(torch.load(f'checkpoint/Classifier_{period}_best_netG_A2B.pth', map_location='cpu'))
    model.eval()

    transforms_list = [ToTensor(), Resize(size_tuple=(256, 256))]
    dataset = EyeDataset(f'data/Classifier/{period}-term', 0,
                         transforms_1=transforms_list, transforms_2=None,
                         unaligned=False, type='test')
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    test_input = torch.Tensor(8, 1, 256, 256)

    real, pred_probs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Inference'):
            bs = batch['A'].size(0)
            inp = Variable(test_input[:bs].copy_(batch['A']))
            _, out = model(inp)
            pred_probs.extend(torch.softmax(out, dim=1).cpu().numpy())
            real.extend(batch['class_label'].numpy())

    real = np.array(real)
    pred = np.array(pred_probs)
    pred_cls = pred.argmax(axis=1)
    real_oh = np.eye(3)[real]

    # Boxplot
    n_bootstrap = 1000
    sen_scores, spe_scores, pre_scores = [], [], []
    for _ in range(n_bootstrap):
        idx = resample(np.arange(len(real)))
        cm = confusion_matrix(real[idx], pred_cls[idx], labels=[0,1,2])
        tp = np.diag(cm); fn = np.sum(cm, axis=1) - tp
        fp = np.sum(cm, axis=0) - tp; tn = np.sum(cm) - tp - fn - fp
        sen_scores.append(np.mean(tp / (tp + fn + 1e-10)))
        spe_scores.append(np.mean(tn / (tn + fp + 1e-10)))
        pre_scores.append(np.mean(tp / (tp + fp + 1e-10)))

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([sen_scores, spe_scores, pre_scores],
                    labels=['Sensitivity', 'Specificity', 'Precision'],
                    patch_artist=True, medianprops={'color':'black','linewidth':2})
    for patch, c in zip(bp['boxes'], ['#4472C4','#ED7D31','#A5A5A5']):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel('Value'); ax.set_ylim([0.75, 1.0])
    ax.set_title(f'{period}-term'); ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(); plt.savefig(f'output/{period}-term/boxplot_metrics.png', dpi=300); plt.close()
    print(f'  Boxplot: Sen={np.mean(sen_scores):.3f}, Spe={np.mean(spe_scores):.3f}, Pre={np.mean(pre_scores):.3f}')

    # ROC + Youden
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['red', 'orange', 'blue']
    for i, c in zip(range(3), colors):
        fpr, tpr, thresholds = roc_curve(real_oh[:,i], pred[:,i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=c, lw=1.5, label=f'{class_names[i]} AUC={roc_auc:.3f}')
        youden = tpr - fpr; best_idx = np.argmax(youden)
        ax.scatter(fpr[best_idx], tpr[best_idx], color=c, s=80, zorder=5)
        print(f'    {class_names[i]}: Youden={youden[best_idx]:.3f} @ threshold={thresholds[best_idx]:.3f}')
    fpr_m, tpr_m, _ = roc_curve(real_oh.ravel(), pred.ravel())
    ax.plot(fpr_m, tpr_m, 'deeppink', lw=1.5, label=f'Micro-avg AUC={auc(fpr_m,tpr_m):.3f}')
    ax.plot([0,1],[0,1],'k--',lw=1); ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(loc='lower right', prop=FontProperties(size='small'))
    ax.set_title(f'{period}-term')
    plt.savefig(f'output/{period}-term/ROC.png', dpi=300, bbox_inches='tight'); plt.close()
    print('  ROC done')

    # PR Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, c in zip(range(3), colors):
        prec_curve, rec_curve, _ = precision_recall_curve(real_oh[:,i], pred[:,i])
        ax.plot(rec_curve, prec_curve, color=c, lw=1.5, label=class_names[i])
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.legend(loc='lower left'); ax.set_title(f'{period}-term')
    plt.savefig(f'output/{period}-term/PR_curve.png', dpi=300, bbox_inches='tight'); plt.close()
    print('  PR done')

    # Per-class metrics
    cm = confusion_matrix(real, pred_cls)
    print(f'  Confusion matrix:\n{cm}')
    for i, name in enumerate(class_names):
        tp = cm[i,i]; fn = np.sum(cm[i,:])-tp; fp = np.sum(cm[:,i])-tp; tn = np.sum(cm)-tp-fn-fp
        sen = tp/(tp+fn) if (tp+fn)>0 else 0
        spe = tn/(tn+fp) if (tn+fp)>0 else 0
        pre = tp/(tp+fp) if (tp+fp)>0 else 0
        print(f'    {name}: Sen={sen:.3f}, Spe={spe:.3f}, Pre={pre:.3f}')

def run_shap(period):
    """SHAP"""
    print(f"\n[SHAP] {period}-term")
    os.makedirs(f'output/{period}-term', exist_ok=True)
    with open('Yaml/CycleGan.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['val_dataroot'] = f'data/Classifier/{period}-term'
    trainer = Cyc_Trainer(cfg)
    trainer.netG_A2B.load_state_dict(
        torch.load(f'checkpoint/Classifier_{period}_best_netG_A2B.pth', map_location='cpu'))
    trainer.netG_A2B.eval()

    class ModelWrapper(torch.nn.Module):
        def __init__(self, gen):
            super().__init__()
            self.gen = gen
        def forward(self, x):
            _, out = self.gen(x); return out
    wrapped = ModelWrapper(trainer.netG_A2B)

    T = transforms.Compose([ToTensor(), Resize(size_tuple=(256, 256))])
    img_dir = f'data/Classifier/{period}-term/before'
    imagename = sorted(os.listdir(img_dir))
    idx = min(166, len(imagename)-9)
    shample = []
    for i in range(idx+1, idx+9):
        img = cv2.imread(os.path.join(img_dir, imagename[i]), 0)
        real_A = trainer.test_input_A.clone().zero_(); real_A[0] = T(img)
        shample.append(real_A.clone())
    shamples = torch.cat(shample, dim=0)
    e = shap.GradientExplainer((wrapped, wrapped.gen.classifier_body[2]), shamples)
    img = cv2.imread(os.path.join(img_dir, imagename[idx]), 0)
    real_A = trainer.test_input_A.clone().zero_(); real_A[0] = T(img)
    shap_values = e.shap_values(real_A, nsamples=200)
    shap_values = [shap_values[i] for i in range(2,0,-1)]
    shap_values = [np.swapaxes(np.swapaxes(s,2,3),1,-1) for s in shap_values]
    for j in range(2):
        shap_values[j][np.where(abs(shap_values[j])<0.01)] = 0
        shap_values[j] = np.rot90(shap_values[j], 3, axes=(1,2))
    real_A_img = real_A.cpu().numpy().squeeze(0).squeeze(0)
    real_A_img = cv2.flip(cv2.transpose(real_A_img), 1).reshape(1,256,256)
    shap.image_plot(shap_values, real_A_img, ['Effective','Ineffective'], show=False)
    plt.savefig(f'output/{period}-term/shap.png', dpi=1200, bbox_inches='tight'); plt.close()
    print(f'  SHAP done')

def run_heatmap(period):
    """Heatmap CAM"""
    print(f"\n[Heatmap] {period}-term")
    os.makedirs(f'output/{period}-term', exist_ok=True)
    model = Generator(1, 1)
    model.load_state_dict(torch.load(f'checkpoint/Classifier_{period}_best_netG_A2B.pth', map_location='cpu'))
    model.eval()
    model_features = torch.nn.Sequential(*list(model.children())[:-3], list(model.children())[-2])
    fc1 = model.state_dict()['classifier_tail.0.weight'].cpu().numpy()
    fc2 = model.state_dict()['classifier_tail.2.weight'].cpu().numpy()
    fc3 = model.state_dict()['classifier_tail.4.weight'].cpu().numpy()
    T = transforms.Compose([ToTensor(), Resize(size_tuple=(256, 256))])
    img_dir = f'data/Classifier/{period}-term/before'
    img_name = sorted(os.listdir(img_dir))[0]
    img = cv2.imread(os.path.join(img_dir, img_name), 0)
    test_input = torch.Tensor(1, 1, 256, 256)
    real_A = Variable(test_input.copy_(T(img).unsqueeze(0)))
    with torch.no_grad():
        _, pre_label = model(real_A)
        pred_class = pre_label.argmax().item()
        features = model_features(real_A).detach().cpu().numpy()
        CAMs = returnCAM(features, fc1, fc2, fc3)
    img_disp = real_A.detach().cpu().squeeze().numpy()
    img_disp = cv2.normalize(img_disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    h, w = img_disp.shape
    img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)
    CAMs[pred_class][CAMs[pred_class] < 150] = 0
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[pred_class], (w, h)), cv2.COLORMAP_JET)
    result = heatmap * 0.5 + img_disp * 0.5
    result = cv2.flip(cv2.transpose(result), 1)
    cv2.imwrite(f'output/{period}-term/heatmap.jpg', result)
    print(f'  Heatmap done')

def run_regression():
    """BCVA"""
    print("\n[Regression] Short-term")
    with open('Yaml/CycleGan.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['dataroot'] = 'data/Regression/Short-term'
    cfg['val_dataroot'] = 'data/Regression/Short-term'
    cfg['save_root'] = 'output/Regression/'
    cfg['batchSize'] = 1; cfg['n_cpu'] = 0
    os.makedirs('output/Regression', exist_ok=True)
    tr = Cyc_Trainer1(cfg)
    tr.netG_A2B.load_state_dict(torch.load('checkpoint/Regression_Short_best_netG_A2B.pth', map_location='cpu'))
    tr.test()
    print('  BCVA done')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_classification', action='store_true')
    parser.add_argument('--skip_shap', action='store_true')
    parser.add_argument('--skip_heatmap', action='store_true')
    parser.add_argument('--skip_regression', action='store_true')
    args = parser.parse_args()

    for p in ['Short', 'Mid', 'Long']:
        if not args.skip_classification:
            run_classification(p)
        if not args.skip_shap:
            run_shap(p)
        if not args.skip_heatmap:
            run_heatmap(p)
    if not args.skip_regression:
        run_regression()
    print('\nAll done!')
