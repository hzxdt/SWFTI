import argparse
import os
from ctypes.wintypes import RECTL

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 仅使用GPU ID=2

import tqdm

parser = argparse.ArgumentParser(description='Export feature matrix of original and reconstructed images')
parser.add_argument('--FR_target', metavar='<FR_target>', type= str, default='ArcFace',
                    help='ArcFace/ElasticFace(FR system from target system)')
parser.add_argument('--path_origin_images', metavar='<path_origin_images>', type= str,
                    help='the path of original images')
parser.add_argument('--path_Reconstructing_images', metavar='<path_Reconstructing_images>', type= str,
                    help='the save suffix of the Reconstructing images')
parser.add_argument('--save_suffix', metavar='<save_suffix>', type= str,
                    help='the save suffix of the Reconstructing images')
parser.add_argument('--batch', metavar='<batch>', type= int, default=32,
                    help='the save suffix of the Reconstructing images')
parser.add_argument('--gpu',metavar='<gpu>', type = int, default=0)
args = parser.parse_args()
# args.path_Reconstructing_images = args.path_Reconstructing_images+f'_{args.FR_target}'
# args.path_Reconstructing_images = args.path_Reconstructing_images+2*f'_{args.FR_target[:3].lower()}'
# args.save_suffix = args.save_suffix+2*f'_{args.FR_target[:3].lower()}'
args.save_suffix = args.save_suffix + '_' + args.FR_target
print(args.save_suffix)
import torch
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
import sys
from src.Dataset import TestDataset_112,TestDataset_1024,TestDataset_256
from torch.utils.data import DataLoader
from scipy.io import loadmat
import numpy as np
import os
import tqdm
ori_fea=[]
rec_fea=[]
Origin_dataset=TestDataset_112(dataset_dir=args.path_origin_images, FR_system= args.FR_target,  device=device)
label=Origin_dataset.labels
Origin_dataloader  = DataLoader(Origin_dataset,  batch_size=args.batch, shuffle=False)
for embedding, _,_ in tqdm.tqdm(Origin_dataloader):
    ori_fea.append(embedding.detach().cpu().numpy())
ori_fea=np.vstack(ori_fea)

if args.path_origin_images==args.path_Reconstructing_images:
    Reconstruct_dataset=TestDataset_112(dataset_dir=args.path_origin_images, FR_system= args.FR_target,  device=device)
elif 'styleswin' in args.path_Reconstructing_images:
    Reconstruct_dataset  = TestDataset_256(dataset_dir=args.path_Reconstructing_images, FR_system= args.FR_target,  device=device)
else:
    Reconstruct_dataset  = TestDataset_1024(dataset_dir=args.path_Reconstructing_images, FR_system= args.FR_target,  device=device)
Reconstruct_dataloader  = DataLoader(Reconstruct_dataset,  batch_size=args.batch, shuffle=False)
for embedding, _,_  in tqdm.tqdm(Reconstruct_dataloader):
    rec_fea.append(embedding.detach().cpu().numpy())
rec_fea=np.vstack(rec_fea)
np.savez(f'./data/feature_npz/metric_data_{args.save_suffix}.npz',recon=np.squeeze(rec_fea),ori=np.squeeze(ori_fea),label=label,batch_idx=0)
print(f'./data/feature_npz/metric_data_{args.save_suffix}.npz')