import sys, os, yaml, torch, cv2, numpy as np
from tqdm import tqdm
sys.path.insert(0, 'E:/projects/Anti-VEGF-0AD6')
from Model.CycleGan import Generator
from trainer.datasets import EyeDataset
from trainer.utils import ToTensor, Resize
from torch.utils.data import DataLoader
from torch.autograd import Variable
from skimage.metrics import structural_similarity as compare_ssim

config = {'input_nc': 1, 'output_nc': 1, 'size': 256}

for period in ['Short', 'Mid', 'Long']:
    print(f"\n{'='*60}\n[Predicted Images] {period}-term\n{'='*60}")
    os.makedirs(f'output/{period}-term/pred', exist_ok=True)
    os.makedirs(f'output/{period}-term/before', exist_ok=True)
    os.makedirs(f'output/{period}-term/after', exist_ok=True)

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

    SSIM, num = 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Generating'):
            real_A = Variable(test_input_A.copy_(batch['A']))
            real_B = Variable(test_input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
            fake_B, _ = model(real_A)
            fake_B = fake_B.detach().cpu().numpy().squeeze()
            real_A_img = real_A.detach().cpu().numpy().squeeze()

            norm_pred = cv2.normalize(fake_B, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
            norm_before = cv2.normalize(real_A_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
            norm_after = cv2.normalize(real_B, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)

            name = batch["name"][0]
            cv2.imwrite(f'output/{period}-term/pred/{name}', norm_pred)
            cv2.imwrite(f'output/{period}-term/before/{name}', norm_before)
            cv2.imwrite(f'output/{period}-term/after/{name}', norm_after)

            SSIM += compare_ssim(fake_B, real_B, data_range=fake_B.max()-fake_B.min())
            num += 1

    print(f'  SSIM: {SSIM/num:.4f}')
    print(f'  Images saved: output/{period}-term/pred/, before/, after/')

print('\nDone!')
