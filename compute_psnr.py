import sys, os, torch, cv2, numpy as np
from tqdm import tqdm
sys.path.insert(0, 'E:/projects/Anti-VEGF-0AD6')
from Model.CycleGan import Generator
from trainer.datasets import EyeDataset
from trainer.utils import ToTensor, Resize
from torch.utils.data import DataLoader
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

config = {'input_nc': 1, 'output_nc': 1, 'size': 256}

for period in ['Short', 'Mid', 'Long']:
    print(f"\n{'='*50}\n[PSNR] {period}-term\n{'='*50}")

    model = Generator(1, 1)
    model.load_state_dict(torch.load(f'checkpoint/Classifier_{period}_best_netG_A2B.pth', map_location='cpu'))
    model.eval()

    transforms_list = [ToTensor(), Resize(size_tuple=(256, 256))]
    dataset = EyeDataset(f'data/Classifier/{period}-term', 0,
                         transforms_1=transforms_list, transforms_2=None,
                         unaligned=False, type='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    test_input_A = torch.Tensor(1, 1, 256, 256)
    test_input_B = torch.Tensor(1, 1, 256, 256)

    psnr_total, num = 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Computing'):
            real_A = Variable(test_input_A.copy_(batch['A']))
            real_B = Variable(test_input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
            fake_B, _ = model(real_A)
            fake_B = fake_B.detach().cpu().numpy().squeeze()

            data_range = max(real_B.max() - real_B.min(), 1e-10)
            psnr_total += compare_psnr(real_B, fake_B, data_range=data_range)
            num += 1

    print(f'PSNR: {psnr_total/num:.4f} dB')

print('\nDone!')
