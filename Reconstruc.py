import argparse

parser = argparse.ArgumentParser(description='Test face reconstruction network——Reconstruct image')
parser.add_argument("--ckpt", type=str, default="models/FFHQ_256.pt")
parser.add_argument("--n_latent", default=14, type=int)#1024----18   256--14
parser.add_argument("--G_channel_multiplier", type=int, default=2)#256<--->2
parser.add_argument("--lr_mlp", default=0.01, type=float, help='Lr mul for 8 * fc')
parser.add_argument("--n_mlp", default=8, type=int)
parser.add_argument("--enable_full_resolution", default=8, type=int, help='Enable full resolution attention index')
parser.add_argument("--use_checkpoint", action="store_true", help='Whether to use checkpoint')
parser.add_argument("--size", type=int, default=256)
parser.add_argument("--style_dim", type=int, default=512)
parser.add_argument('--styleswin_checkpoint', metavar='<path_styleswin_checkpoint>', type= str, default='./stylegan3-r-ffhq-1024x1024.pkl',
                    help='/home/gank/StyleSwin/models/FFHQ_256.pt')
parser.add_argument('--path_LFW_dataset', metavar='<path_LFW_dataset>', type= str, default='/home/gank/NBNet/data/lfw_crop112_adaalign',
                    help='LFW directory`')
parser.add_argument('--FR_system', metavar='<FR_system>', type= str, default='ArcFace',
                    help='ArcFace/ElasticFace/MagFace (FR system from whose database the templates are leaked)')
parser.add_argument('--checkpoint', metavar='<checkpoint>', type= str, default='/home/gank/StyleSwin_git/models/generator.py',
                    help='checkpoint of the new mapping network')
parser.add_argument('--save_suffix', metavar='<save_suffix>', type= str, default='training_styleswin_2layer_id_attr_10_arc_arc',
                    help='the save suffix of the Reconstructing images')
parser.add_argument('--gpu',metavar='<gpu>', type = int, default=6)
args = parser.parse_args()
# args.save_suffix = args.save_suffix+2*f'_{args.FR_system[:3].lower()}'
args.save_suffix=args.save_suffix+f'_{args.FR_system}'
print(args.save_suffix)
import torch
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
print("************ NOTE: The torch device is:", device)
#=================== import Dataset ======================
import sys
from src.Dataset import TestDataset_112
from torch.utils.data import DataLoader

testing_dataset  = TestDataset_112(dataset_dir=args.path_LFW_dataset, FR_system= args.FR_system,  device=device)
test_dataloader  = DataLoader(testing_dataset,  batch_size=1, shuffle=False)

#=================== import Model ======================
from models.generator import Generator
g_ema = Generator(
    args.size, args.style_dim, args.n_mlp, channel_multiplier=args.G_channel_multiplier, lr_mlp=args.lr_mlp,
    enable_full_resolution=args.enable_full_resolution, use_checkpoint=args.use_checkpoint
).to(device)
g_ema.eval()
ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
g_ema.load_state_dict(ckpt["g_ema"])
from src.Network import MappingNetwork
generator = MappingNetwork(z_dim = 8,      # Input latent (Z) dimensionality.
                           c_dim = 512,    # Conditioning label (C) dimensionality, 0 = no labels.
                           w_dim = 512,    # Intermediate latent (W) dimensionality.
                           num_ws = args.n_latent,    # Number of intermediate latents to output.
                           num_layers = 2, # Number of mapping layers.
                            )
generator.load_state_dict(
    torch.load(args.checkpoint, map_location=device, )
)
generator.eval()
generator.to(device)
#=================== Test Model ======================
import random
import numpy as np
import cv2
import os
seed=2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
peo_name=testing_dataset.peo_name
i=0
save_img_dir=f'./data/image/rec_img_lfw_{args.save_suffix}'
for embedding, _,_ in test_dataloader:
    noise = torch.randn(embedding.size(0), generator.z_dim, device=device)
    w = generator(z=noise, c=embedding)
    generated_image = g_ema(noise, latents=w, return_latents=True)[0].detach()
    # generated_image = g_ema(w).detach()
    img = generated_image.squeeze()
    img = (torch.clamp(img, min=-1, max=1) + 1) / 2.0
    im = (img.cpu().detach().numpy().transpose(1, 2, 0))
    im = (im * 255).astype(int)
    peo_dir=os.path.join(save_img_dir, peo_name[i].split('/')[0])
    if not os.path.exists(peo_dir):
        os.makedirs(peo_dir, exist_ok=True)
    cv2.imwrite(f'{os.path.join(save_img_dir, peo_name[i])}',
                np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))
    i=i+1

