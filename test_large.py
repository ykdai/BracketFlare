import os
import argparse
import torch
import torchvision
import numpy as np
from PIL import Image, ImageChops, ImageFilter
from tqdm import tqdm
from torchvision.transforms import transforms, Compose, RandomHorizontalFlip, RandomVerticalFlip
import cv2
from basicsr.archs.mprnet_arch import MPRNet
from basicsr.utils.flare_util import predict_flare_from_6_channel, RandomGammaCorrection

class ImageProcessor:
    # This part is for the image with larger resolution
    def __init__(self, model):
        self.model = model

    def resize_image(self, image, target_size):
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if original_width < original_height:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)

        return image.resize((new_width, new_height))

    def process_image(self, image):
        # Open the original image
        to_tensor=transforms.ToTensor()
        rot_transform = Compose([
        RandomHorizontalFlip(1.0),
        RandomVerticalFlip(1.0)
        ])
        gamma_transform=RandomGammaCorrection(10.0)
        original_image = image

        # Resize the image proportionally to make the shorter side 512 pixels
        resized_image = self.resize_image(original_image, 512)
        rot_resized_image= rot_transform(resized_image)

        # Get the resized image's size
        resized_width, resized_height = resized_image.size

        cropped_image1 = resized_image.crop((0, 0, 512, 512))
        rot_cropped_image1 = rot_resized_image.crop((0, 0, 512, 512))

        cropped_image2 = resized_image.crop((resized_width - 512, resized_height - 512, resized_width, resized_height))
        rot_cropped_image2 = rot_resized_image.crop((resized_width - 512, resized_height - 512, resized_width, resized_height))

        # Convert PIL images to NumPy arrays
        image_array1 = torch.concat((to_tensor(np.array(cropped_image1)),gamma_transform(to_tensor(np.array(rot_cropped_image1)))),0)
        image_array2 = torch.concat((to_tensor(np.array(cropped_image2)),gamma_transform(to_tensor(np.array(rot_cropped_image2)))),0)

        # Process the two cropped images using the model
        processed_image1 = self.model(image_array1.unsqueeze(0).cuda())[0].squeeze(0)
        processed_image2 = self.model(image_array2.unsqueeze(0).cuda())[0].squeeze(0)
        # Apply interpolation to the overlapped region
        if resized_width > 512:
            overlap_width = 512 - (resized_width - 512)
            alpha = torch.linspace(0, 1, steps=overlap_width).view(1, overlap_width, 1).expand(512, overlap_width, 6).permute(2,0,1).cuda()
            merged_image = alpha * processed_image2[:,:, :overlap_width] + (1 - alpha) * processed_image1[:,:, -overlap_width:]
        else:
            overlap_height = 512 - (resized_height - 512)
            alpha = torch.linspace(0, 1, steps=overlap_height).view(overlap_height, 1, 1).expand(overlap_height, 512, 6).permute(2,0,1).cuda()
            merged_image = alpha * processed_image2[:,:overlap_height] + (1 - alpha) * processed_image1[:,-overlap_height:]

        # Concatenate the non-overlapping regions
        if resized_width > 512:
            merged_image = torch.cat((processed_image1[:,:, :512-overlap_width], merged_image, processed_image2[:,:, overlap_width:]), dim=2)
        else:
            merged_image = torch.cat((processed_image1[:,:512-overlap_height], merged_image, processed_image2[:,overlap_height:]), dim=1)

        return merged_image

def inference(input_path, output_path, model_path, inpaint_flag):
    gamma = torch.Tensor([2.2])

    model = MPRNet(img_ch=6, output_ch=6).cuda()
    model.load_state_dict(torch.load(model_path)['params'])
    model.eval()
    processor=ImageProcessor(model)

    input_name_list = os.listdir(input_path)

    os.makedirs(os.path.join(output_path, "deflare_orig"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "deflare"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "flare"), exist_ok=True)

    for cur_input_name in tqdm((input_name_list)):
        torch.cuda.empty_cache()

        cur_input_path = os.path.join(input_path, cur_input_name)
        deflare_orig_path = os.path.join(output_path, "deflare_orig", cur_input_name)
        cur_deflare_path = os.path.join(output_path, "deflare", cur_input_name)
        cur_flare_path = os.path.join(output_path, "flare", cur_input_name)

        cur_input_img = Image.open(cur_input_path).convert("RGB")

        with torch.no_grad():
            output_img = processor.process_image(cur_input_img).unsqueeze(0)
            deflare_img, flare_img_predicted, merge_img_predicted = predict_flare_from_6_channel(output_img, gamma)

            torchvision.utils.save_image(flare_img_predicted, cur_flare_path.replace('png','jpg'))
            torchvision.utils.save_image(deflare_img, cur_deflare_path.replace('png','jpg'))

            deflare_img_np=deflare_img.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            deflare_img_pil=Image.fromarray((deflare_img_np * 255).astype(np.uint8))

            flare_img_orig=ImageChops.difference(cur_input_img.resize(deflare_img_pil.size),deflare_img_pil).resize(cur_input_img.size,resample=Image.BICUBIC)
            flare_mask=flare_img_orig.convert('L') 
            mask = Image.eval(flare_mask, lambda x: 255 if x > 5 else 0)

            if inpaint_flag:
                dilate_pixels = 5
                kernel_size = 2 * dilate_pixels + 1
                mask = mask.filter(ImageFilter.MaxFilter(kernel_size))
                cur_input_array = cv2.cvtColor(np.array(cur_input_img), cv2.COLOR_RGB2BGR)
                mask_array = np.array(mask)
                inpainted_array = cv2.inpaint(cur_input_array, mask_array, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
                inpainted_image = Image.fromarray(cv2.cvtColor(inpainted_array, cv2.COLOR_BGR2RGB))
                inpainted_image.save(deflare_orig_path.replace('png','jpg')) 
            else:
                deflare_img_orig = ImageChops.composite( deflare_img_pil.resize(cur_input_img.size,resample=Image.BICUBIC),cur_input_img, mask)
                deflare_img_orig=ImageChops.difference(cur_input_img,flare_img_orig)
                deflare_img_orig.save(deflare_orig_path.replace('png','jpg'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='test/lq/', 
            help='Input image folder.')
    parser.add_argument('-o', '--output_path', type=str, default='results/', 
            help='Output folder.')
    parser.add_argument('-m', '--model_path', type=str, default='expirements/net_g_last.pth', 
            help='Checkpoint folder.')
    parser.add_argument('--inpaint', action='store_const', const=True, default=False)

    args = parser.parse_args()

    inference(args.input_path, args.output_path, args.model_path, args.inpaint)