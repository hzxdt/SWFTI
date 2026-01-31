import torch
from   torch.utils.data import Dataset
from .loss.FaceIDLoss import get_FaceRecognition_transformer
import torchvision.transforms as transforms
import glob
import random
import numpy as np
import cv2
import os
trans = transforms.Compose([
    transforms.ToTensor(),#0-1,CHW
    transforms.Resize((112, 112)),
])
trans_norm=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112)),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]),
])
image_transforms = transforms.Compose(
    [
        # transforms.ToPILImage(),
        # transforms.Resize((128,128)),
        # transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
def Crop(img,size=112):
    """
    Input:
        - img: RGB or BGR image in 0-1 or 0-255 scale
    Output:
        - new_img: RGB or BGR image in 0-1 or 0-255 scale
    """

    FFHQ_REYE_POS = (480, 380)
    FFHQ_LEYE_POS = (480, 650)

    CROPPED_IMAGE_SIZE=(size, size)
    fixed_positions={'reye': FFHQ_REYE_POS, 'leye': FFHQ_LEYE_POS}

    cropped_positions = {
                        "leye": (51.6, 73.5318),
                        "reye": (51.6, 38.2946)
                         }
    """
    Steps:
        1) find rescale ratio

        2) find corresponding pixel in 1024 image which will be mapped to                          
        the coordinate (0,0) at the croped_and_resized image
        
        3) find corresponding pixel in 1024 image which will be mapped to                          
        the coordinate (112,112) at the croped_and_resized image
        
        4) crop image in 1024
        
        5) resize the cropped image
    """
    # step1: find rescale ratio
    alpha = ( cropped_positions['leye'][1] - cropped_positions['reye'][1] )  /  ( fixed_positions['leye'][1]- fixed_positions['reye'][1] )

    # step2: find corresponding pixel in 1024 image for (0,0) at the croped_and_resized image
    coord_0_0_at_1024 = np.array(fixed_positions['reye']) - 1/alpha* np.array(cropped_positions['reye'])

    # step3: find corresponding pixel in 1024 image for (112,112) at the croped_and_resized image
    coord_112_112_at_1024 = coord_0_0_at_1024 + np.array(CROPPED_IMAGE_SIZE) / alpha

    # step4: crop image in 1024
    cropped_img_1024 = img[int(coord_0_0_at_1024[0]) : int(coord_112_112_at_1024[0]),
                           int(coord_0_0_at_1024[1]) : int(coord_112_112_at_1024[1]),
                           :]

    # step5: resize the cropped image
    resized_and_croped_image = cv2.resize(cropped_img_1024, CROPPED_IMAGE_SIZE)

    return resized_and_croped_image
def Crop256(img,size=112):
    """
     Input:
         - img: RGB or BGR image in 0-1 or 0-255 scale
     Output:
         - new_img: RGB or BGR image in 0-1 or 0-255 scale
     """

    FFHQ_REYE_POS = (120, 95)
    FFHQ_LEYE_POS = (120, 160)

    CROPPED_IMAGE_SIZE = (size, size)
    fixed_positions = {'reye': FFHQ_REYE_POS, 'leye': FFHQ_LEYE_POS}

    cropped_positions = {
        "leye": (51.6, 73.5318),
        "reye": (51.6, 38.2946)
    }
    """
    Steps:
        1) find rescale ratio

        2) find corresponding pixel in 1024 image which will be mapped to                          
        the coordinate (0,0) at the croped_and_resized image

        3) find corresponding pixel in 1024 image which will be mapped to                          
        the coordinate (112,112) at the croped_and_resized image

        4) crop image in 256

        5) resize the cropped image
    """
    # step1: find rescale ratio
    alpha = (cropped_positions['leye'][1] - cropped_positions['reye'][1]) / (
                fixed_positions['leye'][1] - fixed_positions['reye'][1])

    # step2: find corresponding pixel in 1024 image for (0,0) at the croped_and_resized image
    coord_0_0_at_1024 = np.array(fixed_positions['reye']) - 1 / alpha * np.array(cropped_positions['reye'])

    # step3: find corresponding pixel in 1024 image for (112,112) at the croped_and_resized image
    coord_112_112_at_1024 = coord_0_0_at_1024 + np.array(CROPPED_IMAGE_SIZE) / alpha

    # step4: crop image in 1024
    cropped_img_1024 = img[int(coord_0_0_at_1024[0]): int(coord_112_112_at_1024[0]),
                       int(coord_0_0_at_1024[1]): int(coord_112_112_at_1024[1]),
                       :]

    # step5: resize the cropped image
    resized_and_croped_image = cv2.resize(cropped_img_1024, CROPPED_IMAGE_SIZE)

    return resized_and_croped_image
class MyDataset(Dataset):
    def __init__(self, dataset_dir = './Flickr-Faces-HQ/images1024x1024',
                       FR_system= 'ArcFace',
                       train=True,
                       device='cpu',
                       mixID_TrainTest=True,
                       train_test_split = 0.9,
                       random_seed=2021,
                       ori_image=True,
                       no_subdir=True,
                ):
        self.dataset_dir = dataset_dir
        self.device = device
        self.train  = train
        self.FR_system = FR_system
        self.dir_all_images = []
        self.ori_image = ori_image
        # if os.isdir(dataset_dir):
        if no_subdir:
            self.dir_all_images = glob.glob(dataset_dir + '/*.png')
        else:
            all_folders = glob.glob(dataset_dir+'/*')
            all_folders.sort()
            for folder in all_folders:
                all_imgs = glob.glob(folder+'/*.png')
                all_imgs.sort()
                for img in all_imgs:
                    self.dir_all_images.append(img)

        if mixID_TrainTest:
            random.seed(random_seed)
            random.shuffle(self.dir_all_images)

        if self.train:
            self.dir_all_images = self.dir_all_images[:int(train_test_split*len(self.dir_all_images))]
        else:
            self.dir_all_images = self.dir_all_images[int(train_test_split*len(self.dir_all_images)):]

        if FR_system=='ArcFace' or FR_system=='ElasticFace':
            self.Face_Recognition_Network = get_FaceRecognition_transformer(FR=FR_system,device=self.device)

    def __len__(self):
        return len(self.dir_all_images)

    def __getitem__(self, idx):

        image_1024 = cv2.imread(self.dir_all_images[idx]) # (1024, 1024, 3)
        # cv2.imwrite('1.jpg',Crop(image_1024))
        image_HQ = cv2.cvtColor(image_1024, cv2.COLOR_BGR2RGB)
        # image_1024 = cv2.cvtColor(image_1024, cv2.COLOR_BGR2RGB)
        # image_HQ = cv2.resize(image_1024, (256,256))
        if image_HQ.shape[0] == 1024:
            image = Crop(image_HQ) # (112, 112, 3)
        elif image_HQ.shape[0] == 256:
            image = Crop256(image_HQ)
        image256=Crop(image_HQ,256)
        image_HQ = image_HQ/255.
        image    = image/255.
        image256=image256/255.
        image = image.transpose(2,0,1)  # (3, 112, 112)
        image = np.expand_dims(image, axis=0) # (1, 3, 112, 112)
        image256=image256.transpose(2,0,1)
        if self.FR_system == 'ArcFace' or self.FR_system=='ElasticFace':
            img = torch.Tensor( (image*255.).astype('uint8') ).type(torch.FloatTensor)
            embedding = self.Face_Recognition_Network.transform(img.to(self.device) )
        image = image[0] # range (0,1) and shape (3, 112, 112)

        image = self.transform_image(image)
        embedding = self.transform_embedding(embedding)
        embedding_tifs=self.transform_embedding_tifs(embedding)
        image256=self.transform_image(image256)

        image_HQ = image_HQ.transpose(2,0,1)  # (3, 256, 256)
        image_HQ = torch.Tensor( image_HQ ).type(torch.FloatTensor).to(self.device)
        # print(image_HQ.shape)
        # print(image256.shape)
        if self.ori_image:
            return embedding, image, image_HQ,image256
        else:
            return embedding, image,image256

    def transform_image(self,image):
        # image = image/255.
        image = torch.Tensor(image).to(self.device)
        return image

    def transform_embedding(self, embedding):
        # from ipdb import set_trace
        # set_trace()
        embedding = embedding.view(-1).to(self.device)
        return embedding
class TestDataset_112(Dataset):
    def __init__(self, dataset_dir='./data/images/ori_img',
                 FR_system='MagFace',
                 device='cpu',
                 no_subdir=False,
                 random_seed=2021
                 ):
        self.dataset_dir = dataset_dir
        self.device = device
        self.FR_system = FR_system
        self.dir_all_images = []
        # labels=[]
        # print(all_folders)
        labels=[]
        peo_name=[]
        i=1
        if no_subdir:
            all_imgs = glob.glob(dataset_dir + '/*.jpg')
            all_imgs = sorted(all_imgs, key=lambda x: x.lower())
            for img in all_imgs:
                peo_name.append(img.split('/')[-1])
                self.dir_all_images.append(img)
        else:
            all_folders = glob.glob(dataset_dir + '/*')
            all_folders=sorted(all_folders,key=lambda x: x.lower())
            for folder in all_folders:
                all_imgs = glob.glob(folder + '/*.jpg')
                all_imgs=sorted(all_imgs,key=lambda x: x.lower())
                for img in all_imgs:
                    peo_name.append(img.split('/')[-2]+'/'+img.split('/')[-1])
                    labels.append(i)
                    self.dir_all_images.append(img)
                i=i+1
        if FR_system=='ArcFace' or FR_system=='ElasticFace':
            self.Face_Recognition_Network = get_FaceRecognition_transformer(FR=FR_system,device=self.device)
        self.labels=labels
        self.peo_name=peo_name

    def __len__(self):
        return len(self.dir_all_images)

    def __getitem__(self, idx):

        image_1024 = cv2.imread(self.dir_all_images[idx])  # (112, 112, 3)

        image_HQ = cv2.cvtColor(image_1024, cv2.COLOR_BGR2RGB)
        # image_1024 = cv2.cvtColor(image_1024, cv2.COLOR_BGR2RGB)
        # image_HQ = cv2.resize(image_1024, (256,256))

        # image = Crop(image_HQ)  # (112, 112, 3)
        image255 = image_HQ
        # image_HQ = image_HQ / 255.
        image = image255 / 255.
        image255=image255.transpose(2,0,1)
        image255=np.expand_dims(image255,axis=0)
        image = image.transpose(2, 0, 1)  # (3, 112, 112)
        image = np.expand_dims(image, axis=0)  # (1, 3, 112, 112)
        if self.FR_system == 'ArcFace' or self.FR_system=='ElasticFace':
            img = torch.Tensor( (image*255.).astype('uint8') ).type(torch.FloatTensor)
            embedding = self.Face_Recognition_Network.transform(img.to(self.device) )
        # embedding = self.Face_Recognition_Network.transform( (image*255.).astype('uint8') )
        image = image[0]  # range (0,1) and shape (3, 112, 112)

        image = self.transform_image(image)
        embedding = self.transform_embedding(embedding)
        image255 = self.transform_image(image255)
        # image_HQ = image_HQ.transpose(2, 0, 1)  # (3, 256, 256)
        # image_HQ = torch.Tensor(image_HQ).type(torch.FloatTensor).to(self.device)

        # return embedding, image
        return embedding, image,image255

    def transform_image(self, image):
        # image = image/255.
        image = torch.Tensor(image).to(self.device)
        return image

    def transform_embedding(self, embedding):
        # from ipdb import set_trace
        # set_trace()
        embedding = embedding.view(-1).to(self.device)
        return embedding