import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--FR_system", type=str, default='ArcFace')
parser.add_argument("--epoch_to_resume", type=int, default=0)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--path", type=str, default='/home/PublicData/FFHQ/images', help="Path of training data")
parser.add_argument("--size", type=int, default=256)
parser.add_argument("--style_dim", type=int, default=512)
parser.add_argument("--ckpt", type=str, default="models/FFHQ_256.pt")
parser.add_argument("--G_channel_multiplier", type=int, default=2)
parser.add_argument("--lr_mlp", default=0.01, type=float, help='Lr mul for 8 * fc')
parser.add_argument("--n_mlp", default=8, type=int)
parser.add_argument("--enable_full_resolution", default=8, type=int, help='Enable full resolution attention index')
parser.add_argument("--use_checkpoint", action="store_true", help='Whether to use checkpoint')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--weight_attr',type=int, default=10)
args = parser.parse_args()
import torch
from models.generator import Generator
ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
g_ema = Generator(
    args.size, args.style_dim, args.n_mlp, channel_multiplier=args.G_channel_multiplier, lr_mlp=args.lr_mlp,
    enable_full_resolution=args.enable_full_resolution, use_checkpoint=args.use_checkpoint
).to(device)
g_ema.eval()
g_ema.load_state_dict(ckpt["g_ema"])
import os,sys

import pickle
import torch
import torch_utils

import random
import numpy as np
import cv2


seed=args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
from src.Dataset import MyDataset
from torch.utils.data import DataLoader

training_dataset = MyDataset(dataset_dir=args.path, FR_system= args.FR_system, train=True,  device=device)
testing_dataset  = MyDataset(dataset_dir=args.path, FR_system= args.FR_system, train=False, device=device)

train_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)
test_dataloader  = DataLoader(testing_dataset,  batch_size=1, shuffle=False)
# sys.path.append(os.getcwd()) # import src
from src.Network import Discriminator, MappingNetwork
model_Discriminator = Discriminator(dim=14 if args.size==256 else 18)
model_Discriminator.to(device)
model_Generator = MappingNetwork(z_dim = 8,                      # Input latent (Z) dimensionality.
                                 c_dim = 512,                        # Conditioning label (C) dimensionality, 0 = no labels.
                                 w_dim = 512,                      # Intermediate latent (W) dimensionality.
                                 num_ws = 14 if args.size==256 else 18,                      # Number of intermediate latents to output.
                                 num_layers = 2,                   # Number of mapping layers.
                                 )
model_Generator.to(device)
z_dim_Generator = model_Generator.z_dim
z_dim_StyleSwin = 512
from src.loss.FaceIDLoss import ID_Loss
ID_loss = ID_Loss(FR_loss= args.FR_system, device=device)

# ***** Other losses
Pixel_loss = torch.nn.MSELoss()
#**********cosine
def Cos(a,b):
    cos=torch.nn.functional.cosine_similarity(a.view(-1), b.view(-1), dim=0)
    return cos
#**********per
import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
#attr
import torch.nn as nn
sys.path.insert(0, '/home/ljh/teststylegan/mystylegan/BFIRGB')
from FaceAttr.FaceAttrSolver import FaceAttrSolver
attr_model = FaceAttrSolver(epoches=100, batch_size=32, learning_rate=1e-2, model_type='Resnet18', optim_type='SGD',
                            momentum=0.9, pretrained=True, loss_type='BCE_loss', exp_version='v7', device=device)
for param in model_Generator.parameters():
    param.requires_grad = True

# ***** optimizer_Generator
optimizer1_Generator    = torch.optim.Adam(model_Generator.parameters(), lr=1e-1)
scheduler1_Generator    = torch.optim.lr_scheduler.StepLR(optimizer1_Generator, step_size=3, gamma=0.5)

optimizer2_Generator    = torch.optim.Adam(model_Generator.parameters(), lr=1e-1)
scheduler2_Generator    = torch.optim.lr_scheduler.StepLR(optimizer2_Generator, step_size=3, gamma=0.5)
# ***** optimizer_Discriminator
optimizer_Discriminator = torch.optim.Adam(model_Discriminator.parameters(), lr=1e-1)
scheduler_Discriminator = torch.optim.lr_scheduler.StepLR(optimizer_Discriminator, step_size=3, gamma=0.5)
import os

os.makedirs(f'train_seed_{args.seed}_id_attr_{args.weight_attr}', exist_ok=True)
os.makedirs(f'train_seed_{args.seed}_id_attr_{args.weight_attr}/models', exist_ok=True)
os.makedirs(f'train_seed_{args.seed}_id_attr_{args.weight_attr}/Generated_images', exist_ok=True)
os.makedirs(f'train_seed_{args.seed}_id_attr_{args.weight_attr}/logs_train', exist_ok=True)

with open(f'train_seed_{args.seed}_id_attr_{args.weight_attr}/logs_train/generator.csv', 'w') as f:
    f.write("epoch,ID_loss_Gen,Attr_loss_Gen,total_loss,score_Disc_real,score_Disc_fake\n")

with open(f'train_seed_{args.seed}_id_attr_{args.weight_attr}/logs_train/log.txt', 'w') as f:
    pass

for embedding, real_image, real_image_HQ ,image256 in test_dataloader:
    # print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
    # noise = torch.randn(embedding.size(0), z_dim_Generator, device=device)
    # w = model_Generator(z=noise, c=embedding)
    # generated_image, _ = g_ema(noise, latents=w, return_latents=True)
    # print(generated_image.shape)
    # print(real_image.shape)
    # print(generated_image)
    # print(realimage.size())
    # print('ID----------------------------------')
    # ID = ID_loss(generated_image, real_image)
    pass

real_image = real_image.cpu()
real_image_HQ = real_image_HQ.cpu()
for i in range(real_image.size(0)):
    os.makedirs(f'train_seed_{args.seed}_id_attr_{args.weight_attr}/Generated_images/{i}', exist_ok=True)

    img = real_image[i].squeeze()
    im = (img.numpy().transpose(1, 2, 0) * 255).astype(int)
    cv2.imwrite(f'train_seed_{args.seed}_id_attr_{args.weight_attr}/Generated_images/{i}/real_image_cropped.jpg',
                np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))

    img = real_image_HQ[i].squeeze()
    im = (img.numpy().transpose(1, 2, 0) * 255).astype(int)
    cv2.imwrite(f'train_seed_{args.seed}_id_attr_{args.weight_attr}/Generated_images/{i}/real_image_HQ.jpg',
                np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))
# ========================================================
epoch_to_resume = args.epoch_to_resume  # 你想要从哪个epoch开始恢复训练，这个值你需要提前知道
if epoch_to_resume > 0:
    model_Generator.load_state_dict(torch.load(f'train_seed_{args.seed}_id_attr_{args.weight_attr}/models/Generator_{epoch_to_resume}.pth'))
    model_Discriminator.load_state_dict(
        torch.load(f'train_seed_{args.seed}_id_attr_{args.weight_attr}/models/Discriminator_{epoch_to_resume}.pth'))
# =================== Train ==============================
num_epochs = 20 - epoch_to_resume
# =================== Train ==============================
for epoch in range(num_epochs):
    iteration = 0

    print(f'epoch: {epoch + epoch_to_resume}, \t learning rate: {optimizer1_Generator.param_groups[0]["lr"]}')

    for embedding, real_image, real_image_HQ ,image256 in train_dataloader:
        # print(real_image.size)

        if iteration % 4 == 0:
            """
            ******************* GAN: Update Discriminator *******************
            """
            model_Generator.eval()
            model_Discriminator.train()

            # Generate batch of latent vectors
            noise = torch.randn(embedding.size(0), z_dim_Generator, device=device)
            w_fake = model_Generator(z=noise, c=embedding).detach()
            # print(w_fake.size())

            noise = torch.randn(embedding.size(0), z_dim_StyleSwin, device=device)
            _,w_real = g_ema(noise, return_latents=True)
            # ==================forward==================
            # disc should give lower score for real and high for gnerated (fake)
            output_discriminator_real = model_Discriminator(w_real)
            errD_real = output_discriminator_real

            output_discriminator_fake = model_Discriminator(w_fake)
            errD_fake = (-1) * output_discriminator_fake

            loss_GAN_Discriminator = (errD_fake + errD_real).mean()
            W_Pixel=Pixel_loss(w_fake, w_real)
            W_Cosine=-Cos(w_real, w_fake)
            loss_GAN_Discriminator+=W_Pixel+W_Cosine
            # ==================backward=================
            optimizer_Discriminator.zero_grad()
            loss_GAN_Discriminator.backward()
            optimizer_Discriminator.step()

            for param in model_Discriminator.parameters():
                param.data.clamp_(-0.01, 0.01)

        if iteration % 2 == 0:
            model_Generator.train()
            model_Discriminator.eval()
            """
            ******************* GAN: Update Generator *******************
            """
            # Generate batch of latent vectors
            noise = torch.randn(embedding.size(0), z_dim_Generator, device=device)
            w_fake = model_Generator(z=noise, c=embedding)
            # ==================forward==================
            output_discriminator_fake = model_Discriminator(w_fake)

            loss_GAN_Generator = output_discriminator_fake.mean()
            # ==================backward=================
            optimizer1_Generator.zero_grad()
            loss_GAN_Generator.backward()
            optimizer1_Generator.step()

        model_Generator.train()
        """
        ******************* Train Generator *******************
        """
        # ==================forward==================
        noise = torch.randn(embedding.size(0), z_dim_Generator, device=device)
        w = model_Generator(z=noise, c=embedding)
        generated_image,_ = g_ema(noise,latents=w,return_latents=True)
        # print(realimage.size())
        # print('ID----------------------------------')
        ID = ID_loss(generated_image, real_image)
        # Pixel = Pixel_loss((torch.clamp(generated_image, min=-1, max=1) + 1) / 2.0, image256)
        # Per = loss_fn_vgg(generated_image, image256).mean()
        Attr = nn.L1Loss()(attr_model.predict(generated_image), attr_model.predict(image256))
        Attr*=args.weight_attr
        loss_train_Generator = ID+Attr

        # ==================backward=================
        optimizer2_Generator.zero_grad()
        loss_train_Generator.backward()  # (retain_graph=True)
        optimizer2_Generator.step()

        # ==================log======================
        iteration += 1
        if iteration % 200 == 0:
            with open(f'train_seed_{args.seed}_id_attr_{args.weight_attr}/logs_train/log.txt', 'a') as f:
                f.write(
                    f'epoch:{epoch + 1 + epoch_to_resume}, \t iteration: {iteration}, \t loss_train_Generator:{loss_train_Generator.data.item()}, \t loss_GAN_Discriminator:{loss_GAN_Discriminator.data.item()}, \t loss_GAN_Generator:{loss_GAN_Generator.data.item()}\n')
            pass
#     # ******************** Eval Genrator ********************
    model_Generator.eval()
    model_Discriminator.eval()
    ID_loss_Gen_test  =Attr_loss_Gen_test=total_loss_Gen_test = score_Discriminator_fake_test = score_Discriminator_real_test = 0
    iteration = 0
    for embedding, real_image, real_image_HQ ,image256 in test_dataloader:
        # pass
        iteration += 1
        # ==================forward==================
        with torch.no_grad():
            noise = torch.randn(embedding.size(0), z_dim_Generator, device=device)
            w = model_Generator(z=noise, c=embedding)
            generated_image,_ = g_ema(noise,latents=w,return_latents=True)
            ID = ID_loss(generated_image, real_image)
            # Pixel = Pixel_loss((torch.clamp(generated_image, min=-1, max=1) + 1) / 2.0, image256)
            # Per = loss_fn_vgg(generated_image, image256).mean()
            Attr = nn.L1Loss()(attr_model.predict(generated_image), attr_model.predict(image256))
            Attr*=args.weight_attr
            total_loss_Generator = ID+Attr
            ####
            ID_loss_Gen_test += ID.item()
            # Pixel_loss_Gen_test += Pixel.item()
            # Per_loss_Gen_test += Per.item()
            Attr_loss_Gen_test += Attr.item()
            total_loss_Gen_test += total_loss_Generator.item()

            # Eval Discriminator (GAN)
            output_discriminator_fake = model_Discriminator(w)
            score_Discriminator_fake_test += output_discriminator_fake.mean().item()

            noise = torch.randn(embedding.size(0), z_dim_StyleSwin, device=device)
            _,w_real = g_ema(noise,return_latents=True)
            output_discriminator_real = model_Discriminator(w_real)
            score_Discriminator_real_test += output_discriminator_real.mean().item()

    with open(f'train_seed_{args.seed}_id_attr_{args.weight_attr}/logs_train/generator.csv', 'a') as f:
        f.write(
            f"{epoch + 1+epoch_to_resume}, {ID_loss_Gen_test / iteration}, {Attr_loss_Gen_test / iteration},{total_loss_Gen_test / iteration}, {score_Discriminator_real_test / iteration},{score_Discriminator_fake_test / iteration}\n")    #Save model_Generator
    torch.save(model_Generator.state_dict(),
                   f'train_seed_{args.seed}_id_attr_{args.weight_attr}/models/Generator_{epoch + 1 + epoch_to_resume}.pth')
    torch.save(model_Discriminator.state_dict(),
                   f'train_seed_{args.seed}_id_attr_{args.weight_attr}/models/Discriminator_{epoch + 1 + epoch_to_resume}.pth')
    noise = torch.randn(embedding.size(0), z_dim_Generator, device=device)
    w = model_Generator(z=noise, c=embedding)
    generated_image,_ = g_ema(noise,latents=w,return_latents=True)
    for i in range(generated_image.size(0)):
        # img = generated_image[i].squeeze().cpu().detach().numpy()
        # img = (torch.clamp(img, min=-1, max=1) + 1) / 2.0
        # im = (img.cpu().numpy().transpose(1, 2, 0))
        # im = (im * 255).astype(int)
        img = generated_image.squeeze().cpu().detach().numpy()
        img = np.clip(img, -1, 1)  # 限制值范围
        img = (img + 1) / 2.0  # 归一化到[0, 1]
        im = img.transpose(1, 2, 0)  # 转置
        im = (im * 255).astype(np.uint8)  # 转换为图像格式
        cv2.imwrite(f'train_seed_{args.seed}_id_attr_{args.weight_attr}/Generated_images/{i}/epoch_{epoch + 1 + epoch_to_resume}.jpg',
                        np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))
        # *******************************************************

    # Update schedulers
    scheduler1_Generator.step()
    scheduler2_Generator.step()
    scheduler_Discriminator.step()