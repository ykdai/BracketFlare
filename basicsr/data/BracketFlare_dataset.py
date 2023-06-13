import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils import data as data
from torch.distributions import Normal
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from basicsr.utils.registry import DATASET_REGISTRY


class RandomGammaCorrection(object):
	def __init__(self, gamma = None):
		self.gamma = gamma

	def __call__(self,image):
		if self.gamma == None:
			# Make more chances of selecting 0 (original image)
			gammas = [0.5,1,2]
			self.gamma = random.choice(gammas)
			return TF.adjust_gamma(image, self.gamma, gain=1)
		elif isinstance(self.gamma,tuple):
			gamma=random.uniform(*self.gamma)
			return TF.adjust_gamma(image, gamma, gain=1)
		elif self.gamma == 0:
			return image
		else:
			return TF.adjust_gamma(image,self.gamma,gain=1)


class RandomTranslate(object):
	def __init__(self, translate = None):
		self.translate = translate
	def __call__(self, image):
		if self.translate == None:
			self.translate = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))

		translate_base = (
			int(image.shape[1] * self.translate[0]),
			int(image.shape[2] * self.translate[1])
		)
		
		return TF.affine(image, translate=translate_base, angle=0, scale=1, shear=0) , translate_base


def remove_background(image):
	# The input of the image is PIL.Image form with [H,W,C]
	image=np.float32(np.array(image))
	_EPS=1e-7
	rgb_max=np.max(image,(0,1))
	rgb_min=np.min(image,(0,1))
	image=(image-rgb_min)*rgb_max/(rgb_max-rgb_min+_EPS)
	image=torch.from_numpy(image)
	return image


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


@DATASET_REGISTRY.register()
class BracketFlare_Loader(data.Dataset):

	def __init__(self, opt):
		super(BracketFlare_Loader, self).__init__()
		self.opt = opt

		self.ext = ['png','jpeg','jpg','bmp','tif', 'PNG', 'JPEG', 'JPG']
		self.background_list = []
		self.flare_list = []
		for ext in self.ext:
			self.background_list.extend(glob.glob(self.opt['background_path'] + '/*.' + ext))
			self.flare_list.extend(glob.glob(self.opt['flare_path'] + '/*.' + ext))
		self.background_list.sort()
		self.flare_list.sort()
		assert len(self.background_list) == len(self.flare_list), \
			f"Number of backgournd image {len(self.background_list)} is not the same as Number of flare image {len(self.flare_list)}" 

		# mask type is a str which may be None, "luminance" or "color"
		self.mask_type = self.opt['mask_type']
		self.img_size = self.opt['img_size'] if 'img_size' in self.opt else 512
		self.translate = self.opt['translate'] if 'translate' in self.opt else 10/4000
		self.resize_image_size =self.opt['preprocess_size'] if 'preprocess_size' in self.opt else 1000
		self.background_color =self.opt['background_color'] if 'background_color' in self.opt else 0.03
		print("Loaded pair image number:", len(self.background_list))

	def __getitem__(self, index):
		backgournd_img_path = self.background_list[index]
		flare_img_path = self.flare_list[index]
		background_img = Image.open(backgournd_img_path).convert('RGB')
		flare_img = Image.open(flare_img_path).convert('RGB')

		gamma = np.random.uniform(1.8,2.2)
		translate_percentage = (np.random.uniform(-self.translate, self.translate), np.random.uniform(-self.translate, self.translate))
		adjust_gamma = RandomGammaCorrection(gamma)
		adjust_gamma_reverse = RandomGammaCorrection(1/gamma)
		to_tensor = transforms.ToTensor()
		resize = transforms.Resize(self.resize_image_size)
		random_flare = RandomTranslate((translate_percentage))
		color_jitter = transforms.ColorJitter(brightness=(0.8,3), hue=0.0)
		rotate_180 = transforms.Compose([transforms.RandomHorizontalFlip(1), transforms.RandomVerticalFlip(1)])
		blur = transforms.GaussianBlur(21, sigma=(0.1,3.0))
		color_offset_np=np.random.uniform(0,self.background_color,(3,1,1))

		background_img = to_tensor(background_img)
		background_img = resize(background_img)
		flare_img = to_tensor(flare_img)
		flare_img = resize(flare_img)
		flare_img, translate_value = random_flare(flare_img)
		# translate_value (x, y), will be final translate values on lq, while the positive direction is (right, down)
		translate_value = [
			-translate_value[0] * (self.img_size/flare_img.shape[1]),
			-translate_value[1] * (self.img_size/flare_img.shape[2])]

		background_img, flare_img = paired_random_crop(background_img, flare_img, self.img_size, 1)

		background_img = adjust_gamma(background_img)
		sigma_chi = 0.003 * np.random.chisquare(df=1)
		background_img = Normal(background_img,sigma_chi).sample()
		
		color_offset=torch.from_numpy(color_offset_np).float()
		background_img = background_img + color_offset*torch.ones((3,512,512),dtype=torch.float32)
		background_img = torch.clamp(background_img, min=1e-7, max=1) 
		
		flare_img = rotate_180(flare_img)
		flare_img = adjust_gamma(flare_img)
		flare_img = remove_background(flare_img)
		flare_img = color_jitter(flare_img)
		gain = np.random.uniform(0.3,0.7)
		flare_img = torch.clamp(gain*flare_img, min=0, max=1)
		flare_img = blur(flare_img)
		flare_img =  torch.clamp(flare_img, min=0, max=1) 

		merged_img = torch.clamp(flare_img+background_img, min=0, max=1)

		if self.mask_type is None:
			return {
				'gt': adjust_gamma_reverse(background_img), 
				'flare': adjust_gamma_reverse(flare_img),
				'lq': adjust_gamma_reverse(merged_img),
				'gamma': gamma,
				'translate': translate_value,
			}
		else:
			# Calculate mask (the mask is 3 channel)
			if self.mask_type == "luminance":
				one = torch.ones_like(background_img)
				zero = torch.zeros_like(background_img)
				luminance = 0.3*flare_img[0]+0.59*flare_img[1]+0.11*flare_img[2]
				threshold_value = 0.99**gamma
				flare_mask = torch.where(luminance >threshold_value, one, zero)
			elif self.mask_type == "color":
				one = torch.ones_like(background_img)
				zero = torch.zeros_like(background_img)
				threshold_value = 0.99**gamma
				flare_mask = torch.where(merged_img >threshold_value, one, zero)
			elif self.mask_type=="flare":
				one = torch.ones_like(background_img)
				zero = torch.zeros_like(background_img)
				luminance = 0.3*flare_img[0]+0.59*flare_img[1]+0.11*flare_img[2]
				threshold_value=0.03**gamma
				flare_mask=torch.where(luminance >threshold_value, one, zero)
			return {
				'gt': adjust_gamma_reverse(background_img),
				'flare': adjust_gamma_reverse(flare_img),
				'lq': adjust_gamma_reverse(merged_img),
				'gamma': gamma,
				'mask': flare_mask,
				'translate': translate_value,
			}

	def __len__(self):
		return len(self.background_list)


@DATASET_REGISTRY.register()
class Image_Pair_Loader(data.Dataset):
    def __init__(self, opt):
        super(Image_Pair_Loader, self).__init__()
        self.opt = opt
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.paths = glod_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        self.to_tensor=transforms.ToTensor()
        self.gt_size=opt['gt_size']
        self.transform = transforms.Compose([transforms.Resize(self.gt_size), transforms.CenterCrop(self.gt_size), transforms.ToTensor()])

    def __getitem__(self, index):
        gt_path = self.paths['gt'][index]
        lq_path = self.paths['lq'][index]
        img_lq=self.transform(Image.open(lq_path).convert('RGB'))
        img_gt=self.transform(Image.open(gt_path).convert('RGB'))

        return {'lq': img_lq, 'gt': img_gt}

    def __len__(self):
        return len(self.paths)


def glod_from_folder(folder_list, index_list):
	ext = ['png','jpeg','jpg','bmp','tif']
	index_dict={}
	for i,folder_name in enumerate(folder_list):
		data_list=[]
		[data_list.extend(glob.glob(folder_name + '/*.' + e)) for e in ext]
		data_list.sort()
		index_dict[index_list[i]]=data_list
	return index_dict


@DATASET_REGISTRY.register()
class ImageMask_Pair_Loader(Image_Pair_Loader):
    def __init__(self, opt):
        Image_Pair_Loader.__init__(self,opt)
        self.opt = opt
        self.gt_folder, self.lq_folder,self.mask_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_mask']
        self.paths = glod_from_folder([self.lq_folder, self.gt_folder,self.mask_folder], ['lq', 'gt','mask'])
        self.to_tensor=transforms.ToTensor()
        self.gt_size=opt['gt_size']
        self.transform = transforms.Compose([transforms.Resize(self.gt_size), transforms.CenterCrop(self.gt_size), transforms.ToTensor()])

    def __getitem__(self, index):
        gt_path = self.paths['gt'][index]
        lq_path = self.paths['lq'][index]
        mask_path = self.paths['mask'][index]
        img_lq=self.transform(Image.open(lq_path).convert('RGB'))
        img_gt=self.transform(Image.open(gt_path).convert('RGB'))
        img_mask = self.transform(Image.open(mask_path).convert('RGB'))

        return {'lq': img_lq, 'gt': img_gt,'mask':img_mask}

    def __len__(self):
        return len(self.paths)