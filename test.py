import random
from pytorch_msssim import ssim
import numpy as np
import time
import torch
import os
import argparse
import lpips
from dataset import PairLoader,write_img, chw_to_hwc
from utils_basic import AverageMeter, CosineScheduler, pad_img
from torch.utils.data import DataLoader
from collections import OrderedDict
from SSIM_method import SSIM as SSIM_function
from models.Ada4DIR_arch import *
from ptflops import get_model_complexity_info
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# print(torch.cuda.is_available())
# print(torch.version.cuda)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Ada4DIR_d', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--test_set', default='Landsat', type=str, help='test dataset name')#GaoFen Landsat AID
parser.add_argument('--exp', default='Landsat', type=str, help='experiment setting')
args = parser.parse_args()


def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict

def test(test_loader, network, result_dir,degra_type):
	PSNR = AverageMeter()
	SSIM = AverageMeter()
	TIME = AverageMeter()
	LPIPS = AverageMeter()
	loss_fn = lpips.LPIPS(net='alex', spatial=True)
	loss_fn.cuda()

	torch.cuda.empty_cache()

	network.eval()

	os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
	os.makedirs(os.path.join(result_dir, 'res_imgs'), exist_ok=True)
	f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

	for idx, batch in enumerate(test_loader):
		input = batch['source'].cuda() * 2 - 1
		target = batch['target'].cuda() * 2 - 1

		filename = batch['filename'][0]

		with torch.no_grad():
			H, W = input.shape[2:]
			input = pad_img(input, network.patch_size if hasattr(network, 'patch_size') else 16)
			time_start = time.time()
			output = network(input).clamp_(-1, 1)
			time_end = time.time()
			time_c = time_end - time_start  # 运行所花时间

			output = output[:, :, :H, :W]

			# [-1, 1] to [0, 1]
			output = output * 0.5 + 0.5
			target = target * 0.5 + 0.5

			psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

			_, _, H, W = output.size()
			ssim_val = SSIM_function().forward(output, target).mean().item()
			#targetcpu = target.detach().cpu()
			#print(output,target)
			lpips_val = loss_fn.forward(output, target).mean().item()

		PSNR.update(psnr_val)
		SSIM.update(ssim_val)
		TIME.update(time_c)
		LPIPS.update(lpips_val)

		print('Test: [{0}]\t'
			  'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
			  'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})\t'
			  'LPIPS: {lpips.val:.03f} ({lpips.avg:.03f})'
			  .format(idx, psnr=PSNR, ssim=SSIM, lpips=LPIPS))

		f_result.write('%s,%.02f,%.03f,%.03f\n' % (filename, psnr_val, ssim_val, lpips_val))

		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		#write_img(os.path.join(result_dir, 'imgs', filename), out_img)
		res = torch.abs(output-target)
		#print(torch.max(res))
		res = (res * 10).clamp(0,1)
		res_img = chw_to_hwc(res.detach().cpu().squeeze(0).numpy())
		#write_img(os.path.join(result_dir, 'res_imgs', filename), res_img)

	f_result.close()
	print("AVG_Time", TIME.avg)
	print('%.03f | %.04f | %.04f.csv' % (PSNR.avg, SSIM.avg, LPIPS.avg))
	os.rename(os.path.join(result_dir, 'results.csv'),
			  os.path.join(result_dir, '%.03f_%.04f_%.04f.csv' % (PSNR.avg, SSIM.avg, LPIPS.avg)))


def test_single(test_loader, network, result_dir):

	torch.cuda.empty_cache()

	network.eval()

	os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
	f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

	for idx, batch in enumerate(test_loader):
		input = batch['source'].cuda()

		filename = batch['filename'][0]

		with torch.no_grad():
			H, W = input.shape[2:]
			input = pad_img(input, network.patch_size if hasattr(network, 'patch_size') else 16)
			output = network(input).clamp_(-1, 1)
			output = output[:, :, :H, :W]
			# [-1, 1] to [0, 1]
			output = output * 0.5 + 0.5



		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		write_img(os.path.join(result_dir, 'imgs', filename), out_img)

	f_result.close()



def main():
	network = eval(args.model)()
	network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.exp, args.model+'.pth')
	print(saved_model_dir)

	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model+saved_model_dir)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)

	dataset_dir = os.path.join(args.data_dir, args.test_set, 'test')
	degraded_types = ['blur', 'dark', 'haze', 'noise']
	#degraded_types = ['blur_aniso','blur_iso','dark','dark_over','haze','haze_over','noise','noise_5','noise_15','noise_25','noise_50']
	for i in range(len(degraded_types)):
		degraded_type=degraded_types[i]
		if degraded_type=='blur':
			degra_type='deblur'
		elif degraded_type=='dark':
			degra_type='dedark'
		elif degraded_type=='haze':
			degra_type='dehaze'
		elif degraded_type=='noise':
			degra_type='denoise'
		test_dataset = PairLoader(dataset_dir, mode='test', degrade_type=degraded_type)
		test_loader = DataLoader(test_dataset,
								 batch_size=1,
								 num_workers=args.num_workers,
								 pin_memory=True)
		result_dir = os.path.join(args.result_dir, args.test_set, args.exp, args.model, degraded_type)
		test(test_loader, network, result_dir,degra_type)

if __name__ == '__main__':
	main()
