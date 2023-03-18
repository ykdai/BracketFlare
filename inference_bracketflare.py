import os
import argparse
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import transforms, Compose, RandomHorizontalFlip, RandomVerticalFlip

from basicsr.archs.mprnet_arch import MPRNet
from basicsr.utils.flare_util import predict_flare_from_6_channel, RandomGammaCorrection


def inference(input_path, output_path, model_path):
    rot_transform = Compose([
        RandomGammaCorrection(10.0),
        RandomHorizontalFlip(1.0),
        RandomVerticalFlip(1.0)
    ])
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((512,512))
    gamma = torch.Tensor([2.2])

    model = MPRNet(img_ch=6, output_ch=6).cuda()
    model.load_state_dict(torch.load(model_path)['params'])
    model.eval()

    input_name_list = os.listdir(input_path)

    os.makedirs(os.path.join(output_path, "input"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "deflare"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "flare"), exist_ok=True)

    for cur_input_name in tqdm((input_name_list)):
        torch.cuda.empty_cache()

        cur_input_path = os.path.join(input_path, cur_input_name)
        cur_input_save_path = os.path.join(output_path, "input", cur_input_name)
        cur_deflare_path = os.path.join(output_path, "deflare", cur_input_name)
        cur_flare_path = os.path.join(output_path, "flare", cur_input_name)

        cur_input_img = Image.open(cur_input_path).convert("RGB")
        cur_input_img = resize(to_tensor(cur_input_img))
        cur_input_img = cur_input_img.cuda().unsqueeze(0)

        with torch.no_grad():
            lq_rot = rot_transform(cur_input_img)
            lq = torch.concat((cur_input_img, lq_rot),1)
            output_img = model(lq)[0]

            deflare_img, flare_img_predicted, merge_img_predicted = predict_flare_from_6_channel(output_img, gamma)

            torchvision.utils.save_image(cur_input_img, cur_input_save_path)
            torchvision.utils.save_image(flare_img_predicted, cur_flare_path)
            torchvision.utils.save_image(deflare_img, cur_deflare_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='inputs/', 
            help='Input image folder.')
    parser.add_argument('-o', '--output_path', type=str, default='results/', 
            help='Output folder.')
    parser.add_argument('-m', '--model_path', type=str, default='expirements/pretrained_models/mprnet/net_g_last.pth', 
            help='Checkpoint folder.')

    args = parser.parse_args()

    inference(args.input_path, args.output_path, args.model_path)
