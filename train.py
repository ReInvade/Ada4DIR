import random

import numpy as np
import time
import torch
import os
import argparse
import json
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from dataset import Pair4typeLoader, PairLoader
from utils_basic import AverageMeter, CosineScheduler, pad_img
from torch.utils.data import DataLoader, RandomSampler
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from norm_layer import *
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
import vgg16
import math
from SSIM_method import SSIM
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from models.Ada4DIR_arch import *
# os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# print(torch.cuda.is_available())
# print(torch.version.cuda)

def smooth(transMap:torch.tensor):
    m = transMap.size(2)
    n = transMap.size(3)
    sum = 0
    for x in range(1, m - 1):
        for y in range(1, n - 1):
            sum = 0 + abs((transMap[0,0,x + 1, y] - transMap[0,0,x - 1, y]) / 2)*math.pow(math.e,torch.norm((transMap[0,0,x + 1, y] - transMap[0,0,x - 1, y]) / 2))\
                + abs((transMap[0,0,x, y + 1] - transMap[0,0,x, y - 1]) / 2) * math.pow(math.e, torch.norm((transMap[0,0,x, y + 1] - transMap[0,0,x, y - 1]) / 2))
    smooth_result = sum/((m-2)*(n-2))
    return smooth_result

def Dark_prior(img,batchsize):
    w0 = 0.95
    h = img.size()[2]
    w = img.size()[3]
    #print(img.size())
    darkchannel_img,darkchannel_img_indice = torch.max(img,1)
    #print(darkchannel_img.size())
    darkchannel_img = torch.reshape(darkchannel_img, (batchsize, 1, h, w))
    #print(darkchannel_img.size())
    #Air = torch.max(img)
    #t = 1-w0*(darkchannel_img/Air)
    return darkchannel_img  #t

class OHCeLoss(nn.Module):
    def __init__(self):
        super(OHCeLoss,self).__init__()
    def forward(self,pred,onehot_label):
        #print('OHCe',pred.shape,onehot_label)
        pred = pred.squeeze()
        onehot_label = onehot_label.squeeze()
        N = pred.size(0)
        # log_prob = F.log_softmax(pred, dim=1)
        log_prob = torch.log(pred)
        #print('log_prob',log_prob,onehot_label)
        loss = -torch.sum(log_prob * onehot_label) / N
        return loss

def onehot(label: int, classes: int):
    """
    return torch.tensor
    """
    onehot_label = np.zeros([1,classes])
    onehot_label[:,label] = 1
    onehot_label = torch.from_numpy(onehot_label)
    return onehot_label

def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    true_dist = torch.empty(size=label_shape)
    true_dist.fill_(smoothing / (classes - 1))
    _, index = torch.max(true_labels, 1)
    true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)
    return true_dist

def train(train_loader, network, imgtype, criterion, optimizer, scaler, epoch, frozen_bn=False):
    losses = AverageMeter()
    C_MSEs = 0
    C_SSIMs = 0
    bi = 0

    torch.cuda.empty_cache()

    network.eval() if frozen_bn else network.train()
    w = 1
    for batch in train_loader:
        clean_img = batch['target'].cuda()
        blur_img = batch['blur'].cuda()
        noise_img = batch['noise'].cuda()
        hazy_img = batch['hazy'].cuda()
        dark_img = batch['dark'].cuda()
        batchsize, height, width = clean_img.size(0), clean_img.size(2), clean_img.size(3)



        with autocast(args.use_mp):
            classnum = (bi // 16) % 4
            # time_start = time.time()
            de_label = {'deblur': 0, 'denoise': 1, 'dehaze': 2,
                        'dedark': 3, 'clean': 4}  # degra_type
            classes = 4
            label_smooth = 0.1
            if classnum == 0:  # blur
                degraded_img = blur_img
                degraded_type = 'deblur'
            elif classnum == 1:  # noise
                degraded_img = noise_img
                degraded_type = 'denoise'
            elif classnum == 2:  # haze
                degraded_img = hazy_img
                degraded_type = 'dehaze'
            elif classnum == 3:  # dark
                degraded_img = dark_img
                degraded_type = 'dedark'
            degraded_label = onehot(de_label[degraded_type], classes).float()
            degraded_label = smooth_one_hot(degraded_label, classes=classes, smoothing=label_smooth).cuda()
            syn_degraded_img = degraded_img * 2 - 1
            ref_clean_img = clean_img * 2 - 1
            output_list = network(inp_img=syn_degraded_img, degra_type=degraded_type, gt=ref_clean_img, epoch=epoch)
            output = output_list[0]
            l_pix = criterion(output, ref_clean_img) + output_list[1]
            l_total = l_pix
            cri_pred = OHCeLoss().cuda()
            if epoch <= 350:
                l_pred = 0
                for j in range(2, len(output_list)):
                    l_pred = l_pred + cri_pred(output_list[j], degraded_label)
                l_pred = 0.01*torch.sum(l_pred)
                l_total += l_pred

        optimizer.zero_grad()
        scaler.scale(l_total).backward()
        scaler.step(optimizer)
        scaler.update()

        """for i in range(batchsize):
            if (bi*batchsize+i)% 16 == 0:
                save_image(torch.cat((torch.reshape(ref_clean_img[i, :, :, :], (1, 3, height, width)) * 0.5 + 0.5,
                                      torch.reshape(syn_degraded_img[i, :, :, :], (1, 3, height, width)) * 0.5 + 0.5,
                                      torch.reshape(output[i, :, :, :], (1, 3, height, width)) * 0.5 + 0.5), 0),
                           f"train_results/train_degraded2clean_{bi*batchsize+i}.png")"""
        bi = bi + 1
        if args.use_ddp: loss = reduce_mean(l_total, dist.get_world_size())
        losses.update(l_total.item())

    return losses.avg


def valid(val_loader, network, degrade_type, degraded_type):
    PSNR_value = AverageMeter()
    SSIM_value = AverageMeter()
    C_MSEs = 0
    C_SSIMs = 0
    mse = nn.MSELoss()
    torch.cuda.empty_cache()
    bi=0

    network.eval()
    for batch in val_loader:
        degraded_img = batch['source'].cuda()
        clean_img = batch['target'].cuda()
        batchsize, height, width = clean_img.size(0), clean_img.size(2), clean_img.size(3)

        with torch.no_grad():
            syn_degraded_img = degraded_img * 2 - 1
            ref_clean_img = clean_img * 2 - 1
            H, W = syn_degraded_img.shape[2:]
            syn_degraded_img = pad_img(syn_degraded_img, network.module.patch_size if hasattr(network.module, 'patch_size') else 16)
            output = network(inp_img=syn_degraded_img, degra_type=degraded_type)
            output = output.clamp_(-1, 1)
            output = output[:, :, :H, :W]
            loss_supervised_clean_MSE = mse(output * 0.5 + 0.5, ref_clean_img * 0.5 + 0.5)
            C_MSEs += loss_supervised_clean_MSE.mean().item()
            loss_supervised_clean_SSIM = SSIM().forward(output * 0.5 + 0.5, ref_clean_img * 0.5 + 0.5)
            C_SSIMs += loss_supervised_clean_SSIM.mean().item()

        mse_loss = F.mse_loss(output * 0.5 + 0.5, ref_clean_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        ssim = SSIM().forward(output * 0.5 + 0.5, ref_clean_img * 0.5 + 0.5).mean()

        """for i in range(batchsize):
            if (bi*batchsize+i)% 1 == 0:
                save_image(torch.cat((torch.reshape(syn_degraded_img[i, :, :, :], (1, 3, height, width)) * 0.5 + 0.5,
                                      torch.reshape(output[i, :, :, :], (1, 3, height, width)) * 0.5 + 0.5,
                                      torch.reshape(ref_clean_img[i, :, :, :], (1, 3, height, width)) * 0.5 + 0.5), 0),
                           f"test_results/{degrade_type}2clean_{bi*batchsize+i}.png")"""

        SSIM_value.update(ssim.item(), syn_degraded_img.size(0))
        PSNR_value.update(psnr.item(), syn_degraded_img.size(0))

        bi=bi+1

    print(len(val_loader), degrade_type + " val SSIM PSNR", round(SSIM_value.avg, 4), round(PSNR_value.avg, 4))
    return SSIM_value.avg,PSNR_value.avg




parser = argparse.ArgumentParser()
parser.add_argument('--model', default="Ada4DIR_t", type=str, help='model name')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--use_mp', action='store_true', default=False, help='use Mixed Precision')
parser.add_argument('--use_ddp', action='store_true', default=False, help='use Distributed Data Parallel')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--train_set', default='Landsat/train', type=str, help='train dataset name')
parser.add_argument('--val_set', default='Landsat/val', type=str, help='valid dataset name')
parser.add_argument('--exp', default='Landsat', type=str, help='experiment setting')
parser.add_argument('--imgtype', default='Natural_Color', type=str, help='image band type')
args = parser.parse_args()

# training environment
if args.use_ddp:
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    if local_rank == 0: print('==> Using DDP.')
else:
    world_size = 1

# training config
with open(os.path.join('configs', args.exp, 'base.json'), 'r') as f:
    b_setup = json.load(f)

variant = args.model.split('_')[-1]
config_name = 'model_' + variant + '.json' if variant in ['t', 's', 'b',
                                                          'd'] else 'default.json'  # default.json as baselines' configuration file
with open(os.path.join('configs', args.exp, config_name), 'r') as f:
    m_setup = json.load(f)


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    print("GPU_available:", torch.cuda.is_available())
    print("GPU_num:", torch.cuda.device_count())
    # define network, and use DDP for faster training
    network = eval(args.model)()
    network.cuda()
    imgtype = args.imgtype

    if args.use_ddp:
        print("Use DDP!")
        network = DistributedDataParallel(network, device_ids=[local_rank], output_device=local_rank)
        if m_setup['batch_size'] // world_size < 16:
            if local_rank == 0: print('==> Using SyncBN because of too small norm-batch-size.')
            nn.SyncBatchNorm.convert_sync_batchnorm(network)
    else:
        print("Use DP!")
        network = DataParallel(network, device_ids=[0])#torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        if m_setup['batch_size'] // torch.cuda.device_count() < 16:
            print('==> Using SyncBN because of too small norm-batch-size.')
            convert_model(network)
    # define loss function
    criterion = nn.L1Loss()
    # define optimizer
    optimizer = torch.optim.AdamW(network.parameters(), lr=m_setup['lr'], weight_decay=b_setup['weight_decay'])
    lr_scheduler = CosineScheduler(optimizer, param_name='lr', t_max=b_setup['epochs'], value_min=m_setup['lr'] * 1e-2,
                                   warmup_t=b_setup['warmup_epochs'], const_t=b_setup['const_epochs'])
    wd_scheduler = CosineScheduler(optimizer, param_name='weight_decay', t_max=b_setup['epochs'])  # seems not to work
    scaler = GradScaler()

    # load saved model
    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(os.path.join(save_dir, args.model + '.pth')):
        best_psnr = 0
        cur_epoch = 0
    else:
        if not args.use_ddp or local_rank == 0: print('==> Loaded existing trained model.')
        model_info = torch.load(os.path.join(save_dir, args.model + '.pth'), map_location='cpu')
        network.load_state_dict(model_info['state_dict'])
        optimizer.load_state_dict(model_info['optimizer'])
        lr_scheduler.load_state_dict(model_info['lr_scheduler'])
        wd_scheduler.load_state_dict(model_info['wd_scheduler'])
        scaler.load_state_dict(model_info['scaler'])
        cur_epoch = model_info['cur_epoch']
        best_psnr = model_info['best_psnr']

    # define dataset
    print(args.train_set)
    train_dataset = Pair4typeLoader(os.path.join(args.data_dir, args.train_set), 'train', 'landsat8trans',
                               b_setup['t_patch_size'],
                               b_setup['edge_decay'],
                               b_setup['data_augment'],
                               b_setup['cache_memory'])
    print("batchsize", m_setup['batch_size'] // (world_size))
    train_loader = DataLoader(train_dataset,
                              batch_size=m_setup['batch_size'] // (world_size),
                              sampler=RandomSampler(train_dataset, num_samples=b_setup['num_iter'] // (world_size)),
                              #shuffle=True,
                              num_workers=args.num_workers // (world_size),
                              pin_memory=True,
                              drop_last=True,
                              persistent_workers=True)  # comment this line for cache_memory
    val_blur_dataset = PairLoader(os.path.join(args.data_dir, args.val_set), b_setup['valid_mode'], 'blur',
                             b_setup['v_patch_size'])
    val_blur_loader = DataLoader(val_blur_dataset,
                            batch_size= 1,
                            num_workers=args.num_workers // (world_size),
                            pin_memory=True)
    val_noise_dataset = PairLoader(os.path.join(args.data_dir, args.val_set), b_setup['valid_mode'],
                                          'noise',
                                          b_setup['v_patch_size'])
    val_noise_loader = DataLoader(val_noise_dataset,
                                 batch_size=1,
                                 num_workers=args.num_workers // (world_size),
                                 pin_memory=True)
    val_haze_dataset = PairLoader(os.path.join(args.data_dir, args.val_set), b_setup['valid_mode'],
                                          'haze',
                                          b_setup['v_patch_size'])
    val_haze_loader = DataLoader(val_haze_dataset,
                                 batch_size=1,
                                 num_workers=args.num_workers // (world_size),
                                 pin_memory=True)
    val_dark_dataset = PairLoader(os.path.join(args.data_dir, args.val_set), b_setup['valid_mode'],
                                          'dark',
                                          b_setup['v_patch_size'])
    val_dark_loader = DataLoader(val_dark_dataset,
                                 batch_size=1,
                                 num_workers=args.num_workers // (world_size),
                                 pin_memory=True)

    # start training
    if not args.use_ddp or local_rank == 0:
        print('==> Start training, current model name: ' + args.model)
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

    for epoch in tqdm(range(cur_epoch, b_setup['epochs'] + 1)):
        frozen_bn = epoch > (b_setup['epochs'] - b_setup['frozen_epochs'])
        loss = train(train_loader, network, imgtype, criterion, optimizer, scaler, epoch, frozen_bn)
        lr_scheduler.step(epoch + 1)
        wd_scheduler.step(epoch + 1)

        if not args.use_ddp or local_rank == 0:
            writer.add_scalar('train_loss', loss, epoch)

        if epoch % b_setup['eval_freq'] == 0:
            b_avg_ssim, b_avg_psnr = valid(val_blur_loader, network, "blur", 'deblur')
            writer.add_scalar('Blur_SSIM', b_avg_ssim, epoch)
            writer.add_scalar('Blur_PSNR', b_avg_psnr, epoch)
            n_avg_ssim, n_avg_psnr = valid(val_noise_loader, network, "noise", 'denoise')
            writer.add_scalar('Noise_SSIM', n_avg_ssim, epoch)
            writer.add_scalar('Noise_PSNR', n_avg_psnr, epoch)
            h_avg_ssim, h_avg_psnr = valid(val_haze_loader, network, "haze", 'dehaze')
            writer.add_scalar('Haze_SSIM', h_avg_ssim, epoch)
            writer.add_scalar('Haze_PSNR', h_avg_psnr, epoch)
            d_avg_ssim, d_avg_psnr = valid(val_dark_loader, network, "dark", 'dedark')
            writer.add_scalar('Dark_SSIM', d_avg_ssim, epoch)
            writer.add_scalar('Dark_PSNR', d_avg_psnr, epoch)
            avg_psnr = (b_avg_psnr+n_avg_psnr+h_avg_psnr+d_avg_psnr)/4

            if not args.use_ddp or local_rank == 0:
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'cur_epoch': epoch + 1,
                                'best_psnr': best_psnr,
                                'state_dict': network.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'wd_scheduler': wd_scheduler.state_dict(),
                                'scaler': scaler.state_dict()},
                               os.path.join(save_dir, args.model + '.pth'))

                writer.add_scalar('valid_psnr', avg_psnr, epoch)
                writer.add_scalar('best_psnr', best_psnr, epoch)

            if args.use_ddp: dist.barrier()


if __name__ == '__main__':
    main()
