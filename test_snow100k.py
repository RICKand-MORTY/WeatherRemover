import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data_functions import ValData
from metrics import calculate_psnr, calculate_ssim
import os
import numpy as np
import random
from cmformer import CMFormer
import torchvision.utils as tvu
import cv2

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-save_place', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-checkpoint', help='select checkpoint of model', type=str)
args = parser.parse_args()

val_batch_size = args.val_batch_size


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)


# set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))

# --- Set category-specific hyper-parameters  --- #
val_data_dir = '../TransWeather/data/snow100k/test_all/Snow100K-L/'

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# --- Validation data loader --- #

val_filename = 'test_all.txt'  ## This text file should contain all the names of the images and must be placed in ./data/test/ directory

val_data_loader = DataLoader(ValData(val_data_dir, val_filename), batch_size=val_batch_size, shuffle=False,
                             num_workers=1)


# --- Define the network --- #

net = CMFormer()
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

# --- Load the network weight --- #
net.load_state_dict(torch.load(args.checkpoint))
total = sum([param.nelement() for param in net.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))
# --- Use the evaluation model in testing --- #
net.eval()
category = "snowtest100k"

if os.path.exists(args.save_place) == False:
    os.makedirs(args.save_place)
    os.makedirs(os.path.join(args.save_place, 'data'))
    os.makedirs(os.path.join(args.save_place, 'gt'))

count = 0
total_time = 0
print('--- Testing starts! ---')
with torch.no_grad():
    for train_data in val_data_loader:
            input_image, gt, imgid = train_data
            input_image = input_image.to(device)
            gt = gt.to(device)
            start_time = time.time()
            result_img = net(input_image)
            end_time = time.time() - start_time
            print(f"process {count}.png, spend {end_time}")
            save_image(result_img, os.path.join(os.path.join(args.save_place, 'data'), '{}.png'.format(count)))
            save_image(gt, os.path.join(os.path.join(args.save_place, 'gt'), '{}.png'.format(count)))
            count += 1
            total_time += end_time

print('average validation time is {0:.4f}, total image is {1}'.format(total_time / count, count))

cumulative_psnr, cumulative_ssim = 0, 0

print('--- Evaluating! ---')
for i in range(count):
    res = cv2.imread(os.path.join(os.path.join(args.save_place, 'data'), '{}.png'.format(i)), cv2.IMREAD_COLOR)
    gt = cv2.imread(os.path.join(os.path.join(args.save_place, 'gt'), '{}.png'.format(i)), cv2.IMREAD_COLOR)
    cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
    cur_ssim = calculate_ssim(res, gt, test_y_channel=True)
    print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
    cumulative_psnr += cur_psnr
    cumulative_ssim += cur_ssim

print('psnr_avg: {0:.4f}, ssim_avg: {1:.4f}'.format(cumulative_psnr / count, cumulative_ssim / count))

with open(os.path.join(args.save_place, "log.txt"), "w") as f:
    f.write('psnr_avg: {0:.4f}, ssim_avg: {1:.4f}\n'.format(cumulative_psnr / count, cumulative_ssim / count))
    f.write('average validation time is {0:.4f}, total image is {1}'.format(total_time / count, count))
    f.close()
