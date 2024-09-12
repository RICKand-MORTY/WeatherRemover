import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.train_data_functions import TrainData
from utils.val_data_functions import ValData
from utils.metrics import calculate_psnr, calculate_ssim
import os
import numpy as np
import random
import torchvision.utils as tvu
import cv2
from model.cmformer import CMFormer


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[128, 128], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=18, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
#parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default=200, type=int)
parser.add_argument('-num_steps', help='number of epochs', default=90000, type=int)
parser.add_argument('-checkpoint', help='resume checkpoint', type=str)
parser.add_argument('-save_epoch', help='save per epoch', default=10, type=int)
parser.add_argument('-save_step', help='save per step', default=1000, type=int)
args = parser.parse_args()

num_steps = args.num_steps
save_step = args.save_step
learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
#lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs
save_epoch = args.save_epoch

#create directory to save checkpoints
if not os.path.exists(exp_name):
    os.makedirs(exp_name, exist_ok=True)

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

print('--- Hyper-parameters for training ---')
print(
    'learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\n'.format(learning_rate,
                                                                                                         crop_size,
                                                                                                         train_batch_size,
                                                                                                         val_batch_size,))

train_data_dir = './data/allweather/train/'
val_data_dir = './data/allweather/test/'

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
net = CMFormer()
total = sum([param.nelement() for param in net.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))

# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

# --- Load the network weight --- #
chk = args.checkpoint
try:
    net.load_state_dict(torch.load(chk))
    print('--- weight loaded ---')
except:
    raise FileNotFoundError(f"The file at path '{chk}' does not exist.")


# --- Load training data and validation/test data --- #

labeled_name = 'train.txt'


val_filename1 = 'test_all.txt'

# --- Load training data and validation/test data --- #
lbl_train_data_loader = DataLoader(TrainData(crop_size, train_data_dir, labeled_name, random_flip=True, random_rotate=True), batch_size=train_batch_size,
                                   shuffle=True, num_workers=8)


# val_data_loader = DataLoader(ValData(val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False, num_workers=8)
val_data_loader1 = DataLoader(ValData(val_data_dir, val_filename1), batch_size=val_batch_size, shuffle=False,
                              num_workers=8)


def hub_loss(img, gt):
    c = 0.03
    diff = torch.sqrt(torch.pow(img - gt, 2) + c ** 2)
    loss = diff - c
    loss = loss.sum() / loss.numel()
    return loss

file_path = "./eva/eva.txt"
total_steps = 0
net.train()
if chk:
    total_steps = int(chk.split("/")[-1].split("_")[0])
while True:
    for batch_id, train_data in enumerate(lbl_train_data_loader):

        input_image, gt, imgid = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        pred_image = net(input_image)

        loss = hub_loss(pred_image, gt)

        loss.backward()
        optimizer.step()
        total_steps += 1

        if (total_steps % 10) == 0:
            print('Steps: {0}, loss: {1}'.format(total_steps, loss))
        if (total_steps % 100) == 0:
            with torch.no_grad():
                save_image(pred_image, os.path.join("./train_res", f"output.png"))
                save_image(gt, os.path.join("./train_res", f"gt.png"))
                save_image(input_image, os.path.join("./train_res", f"input.png"))
        # --- Test --- #
        if total_steps % save_step == 0:
            net.eval()
            i = 0
            with torch.no_grad():
                start_time = time.time()
                for batch_id, test_data in enumerate(val_data_loader1):
                    input_image, gt, imgid = test_data
                    input_image = input_image.to(device)
                    gt = gt.to(device)
                    save_image(gt, os.path.join("./eva/gt", f"{i}.png"))
                    pred_image = net(input_image)
                    save_image(pred_image, os.path.join("./eva/output", f"{i}.png"))
                    i += 1
                test_time = time.time() - start_time
                per_time = test_time / i
                print(f"test speed: {per_time} per image")
                results_path = "./eva/output"
                gt_path = "./eva/gt"
                imgsName = sorted(os.listdir(results_path))
                gtsName = sorted(os.listdir(gt_path))
                assert len(imgsName) == len(gtsName)
    
                cumulative_psnr, cumulative_ssim = 0, 0
                for i in range(len(imgsName)):
                    # logger('Processing image: %s' % (imgsName[i]))
                    res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
                    gt = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)
                    # logger(f"image:{imgsName[i]}, gt:{gtsName[i]}")
                    cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
                    cur_ssim = calculate_ssim(res, gt, test_y_channel=True)
                    # logger('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
                    cumulative_psnr += cur_psnr
                    cumulative_ssim += cur_ssim
                print('Testing set, PSNR is %.4f and SSIM is %.4f' % (
                    cumulative_psnr / len(imgsName), cumulative_ssim / len(imgsName)))
                psnr = cumulative_psnr / len(imgsName)
                ssim = cumulative_ssim / len(imgsName)
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as file:
                        file.write(f"steps:{total_steps}, PSNR:{psnr}, SSIM:{ssim}\n")
                else:
                    with open(file_path, 'a') as file:
                        file.write(f"steps:{total_steps}, PSNR:{psnr}, SSIM:{ssim}\n")
                # --- Save the network parameters --- #
                torch.save(net.state_dict(), './{}/{}_ckpt'.format(exp_name, total_steps))
            torch.cuda.empty_cache()
        if total_steps == num_steps:
            print("Finish!")
            exit()