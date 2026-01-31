import torch
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
import numpy as np
import imp
import os
from bob.extension.download import get_file
import sys
import torchvision.transforms as transforms
trans_norm = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Resize((112, 112)),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]),
])
trans = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Resize((112, 112)),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]),
])

def Crop_and_resize(img):

    FFHQ_REYE_POS = (480, 380) 
    FFHQ_LEYE_POS = (480, 650) 
    
    CROPPED_IMAGE_SIZE=(112, 112)
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
    # cropped_img_1024 = img[int(coord_0_0_at_1024[0])     : int(coord_0_0_at_1024[1]),
    #                        int(coord_112_112_at_1024[0]) : int(coord_112_112_at_1024[1]),
    #                        :]
    cropped_img_1024 = img[:,
                           :,
                           int(coord_0_0_at_1024[0]) : int(coord_112_112_at_1024[0]),
                           int(coord_0_0_at_1024[1]) : int(coord_112_112_at_1024[1])
                           ]
    
    # step5: resize the cropped image
    # resized_and_croped_image = cv2.resize(cropped_img_1024, CROPPED_IMAGE_SIZE) 
    resized_and_croped_image = torch.nn.functional.interpolate(cropped_img_1024, mode='bilinear', size=CROPPED_IMAGE_SIZE, align_corners=False)

    return resized_and_croped_image


def Crop_and_resize_256(img):
    FFHQ_REYE_POS = (120, 95)
    FFHQ_LEYE_POS = (120, 160)

    CROPPED_IMAGE_SIZE = (112, 112)
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

        4) crop image in 1024

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
    # cropped_img_1024 = img[int(coord_0_0_at_1024[0])     : int(coord_0_0_at_1024[1]),
    #                        int(coord_112_112_at_1024[0]) : int(coord_112_112_at_1024[1]),
    #                        :]
    cropped_img_1024 = img[:,
                       :,
                       int(coord_0_0_at_1024[0]): int(coord_112_112_at_1024[0]),
                       int(coord_0_0_at_1024[1]): int(coord_112_112_at_1024[1])
                       ]

    # step5: resize the cropped image
    # resized_and_croped_image = cv2.resize(cropped_img_1024, CROPPED_IMAGE_SIZE)
    resized_and_croped_image = torch.nn.functional.interpolate(cropped_img_1024, mode='bilinear',
                                                               size=CROPPED_IMAGE_SIZE, align_corners=False)

    return resized_and_croped_image
class PyTorchModel(TransformerMixin, BaseEstimator):
    """
    Base Transformer using pytorch models


    Parameters
    ----------

    checkpoint_path: str
       Path containing the checkpoint

    config:
        Path containing some configuration file (e.g. .json, .prototxt)

    preprocessor:
        A function that will transform the data right before forward. The default transformation is `X/255`

    """

    def __init__(
        self,
        checkpoint_path=None,
        config=None,
        preprocessor=lambda x: (x - 127.5) / 128.0,
        device='cpu',
        **kwargs
    ):

        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.model = None
        self.preprocessor_ = preprocessor
        self.device=device

    def preprocessor(self, X):
        X = self.preprocessor_(X)
        if X.size(2) != 112:
            if X.size(2)==1024:
                X = Crop_and_resize(X)
            elif X.size(2)==256:
                X = Crop_and_resize_256(X)
        return X
        
    def transform(self, X):
        """__call__(image) -> feature

        Extracts the features from the given image.

        **Parameters:**

        image : 2D :py:class:`numpy.ndarray` (floats)
        The image to extract the features from.

        **Returns:**

        feature : 2D or 3D :py:class:`numpy.ndarray` (floats)
        The list of features extracted from the image.
        """
        if self.model is None:
            self._load_model()
            
            self.model.eval()
            
            self.model.to(self.device)
            for param in self.model.parameters():
                param.requires_grad=False
                
        # X = check_array(X, allow_nd=True)
        # X = torch.Tensor(X)
        X = self.preprocessor(X)
        # print(X.shape)
        # print(X)
        return self.model(X)#.detach().numpy()


    def __getstate__(self):
        # Handling unpicklable objects

        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}
    
    def to(self,device):
        self.device=device
        
        if self.model !=None:            
            self.model.to(self.device)



def _get_iresnet_file():
    urls = [
        "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet-91a5de61.tar.gz",
        "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet-91a5de61.tar.gz",
    ]

    return get_file(
        "iresnet-91a5de61.tar.gz",
        urls,
        cache_subdir="data/pytorch/iresnet-91a5de61/",
        file_hash="3976c0a539811d888ef5b6217e5de425",
        extract=True,
    )

class IResnet100Elastic(PyTorchModel):
    """
    ElasticFace model
    """

    def __init__(self,  
                preprocessor=lambda x: (x - 127.5) / 128.0, 
                device='cpu'
                ):

        self.device = device
        
        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet100-elastic.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet100-elastic.tar.gz",
        ]

        filename= get_file(
            "iresnet100-elastic.tar.gz",
            urls,
            cache_subdir="data/pytorch/iresnet100-elastic/",
            file_hash="0ac36db3f0f94930993afdb27faa4f02",
            extract=True,
        )

        path = os.path.dirname(filename)
        config = os.path.join(path, "iresnet.py")
        checkpoint_path = os.path.join(path, "iresnet100-elastic.pt")

        super(IResnet100Elastic, self).__init__(
            checkpoint_path, config, device=device,  preprocessor=preprocessor, 
        )

    def _load_model(self):

        model = imp.load_source("module", self.config).iresnet100(self.checkpoint_path)
        self.model = model
        


def _get_iresnet_file():
    urls = [
        "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet-91a5de61.tar.gz",
        "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet-91a5de61.tar.gz",
    ]

    return get_file(
        "iresnet-91a5de61.tar.gz",
        urls,
        cache_subdir="data/pytorch/iresnet-91a5de61/",
        file_hash="3976c0a539811d888ef5b6217e5de425",
        extract=True,
    )

class IResnet100(PyTorchModel):
    """
    ArcFace model (RESNET 100) from Insightface ported to pytorch
    """

    def __init__(self,  
                preprocessor=lambda x: (x - 127.5) / 128.0, 
                device='cpu'
                ):

        self.device = device
        filename = _get_iresnet_file()
        #
        path = os.path.dirname(filename)
        # path='/home/gank/bob_data/data/pytorch/iresnet-91a5de61'
        # path='/data2/gkk/bob_data/data/pytorch/iresnet-91a5de61'
        config = os.path.join(path, "iresnet.py")
        checkpoint_path = os.path.join(path, "iresnet100-73e07ba7.pth")

        super(IResnet100, self).__init__(
            checkpoint_path, config, device=device
        )

    def _load_model(self):

        model = imp.load_source("module", self.config).iresnet100(self.checkpoint_path)
        self.model = model


def get_FaceRecognition_transformer(FR='ArcFace', device='cpu'):
    if FR== 'ArcFace':
        FaceRecognition_transformer = IResnet100(device=device)
    elif FR== 'ElasticFace':
        FaceRecognition_transformer = IResnet100Elastic(device=device)
    else:
        print(f"[FaceIDLoss] {FR} is not defined!")
    return FaceRecognition_transformer 

class ID_Loss:
    def __init__(self, FR_loss='ArcFace', device='cpu' ):
        self.FR_loss = FR_loss
        self.device = device
        self.FR_system=self.FR_loss
        if FR_loss == 'ArcFace' or FR_loss == 'ElasticFace':
            self.Face_Recognition_Network = get_FaceRecognition_transformer(FR=FR_loss, device=device)
    def __call__(self, img1,img2):
        """
        img1: generated     range: (-1,+1) +- delta
        img2: real          range: (0,1)
        """
        # img1=alignMain(img1)
        img1 = torch.clamp(img1, min=-1, max=1)
        img1 = (img1 + 1) / 2.0 # range: (0,1)
        # # print(img1.shape)
        # x=img1
        # x = x[:, :, 35:223, 32:220]
        # x=torch.nn.AdaptiveAvgPool2d((112, 112))(x)
        # for batch in range(img1.shape[0]):
        #     img1[batch]=torch.(img1[batch])
        # img2 = (img2- img2.min(axis=0))/ img2.max(axis=0)
        # print(img2.shape)
        # print(img2)
        # embedding1=self.FaceRecognition_transformer.transform(img1)
        if self.FR_system == 'ArcFace' or self.FR_system == 'ElasticFace':
            # img1 = torch.Tensor((img1 * 255.)).type(torch.FloatTensor)
            # img2 = torch.Tensor((img2 * 255.)).type(torch.FloatTensor)
            embedding1 = self.Face_Recognition_Network.transform(img1 * 255.)
            embedding2 = self.Face_Recognition_Network.transform(img2 * 255.)
        return torch.nn.MSELoss()(embedding1, embedding2)