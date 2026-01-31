import FaceAttr.config as cfg
import torch
from FaceAttr.FaceAttr_baseline_model import FaceAttrModel
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F


class FaceAttrSolver(object):
    
    def __init__(self, epoches, batch_size, learning_rate, model_type, 
        optim_type, momentum, pretrained, loss_type, exp_version,device):

        self.epoches = epoches 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.selected_attrs = cfg.selected_attrs
        self.momentum = momentum
        self.device = device
        self.image_dir = cfg.image_dir
        self.attr_path = cfg.attr_path
        self.pretrained = pretrained
        self.model_type = model_type
        self.build_model(model_type, pretrained)
        self.create_optim(optim_type)
        self.train_loader = None
        self.validate_loader = None
        self.test_loader = None
        self.log_dir = cfg.log_dir
        self.use_tensorboard = cfg.use_tensorboard
        self.attr_loss_weight = torch.tensor(cfg.attr_loss_weight).to(self.device)
        self.attr_threshold = cfg.attr_threshold
        self.model_save_path = None
        self.LOADED = False
        self.start_time = 0
        self.loss_type = loss_type
        self.exp_version = exp_version
        # torch.cuda.set_device(cfg.DEVICE_ID)
        self.set_transform("predict")
        if not self.LOADED:
            # load the best model dict.
            tmp = torch.load("FaceAttr/FaceAttrResnet18.pth",map_location=self.device)
            # tmp.pop('featureClassfier.fc.6.weight')
            # tmp.pop('featureClassfier.fc.6.bias')
            self.model.load_state_dict(tmp)
            # self.model = nn.Sequential(*list(self.model.children())[:-2]) # list(list(self.model.children())[0].children())+list(list(list(self.model.children())[1].children())[0][:-1])
            self.LOADED = True

    def build_model(self, model_type, pretrained):
        """Here should change the model's structure""" 
        self.model = FaceAttrModel(model_type, pretrained, self.selected_attrs).to(self.device).to(self.device)
        
    def create_optim(self, optim_type):
        scheduler = None
        if optim_type == "Adam":
            self.optim_ = optim.Adam(self.model.parameters(), lr = self.learning_rate)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim_, [30,80], gamma=0.1)
        elif optim_type == "SGD":
            self.optim_ = optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum = self.momentum)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim_, [30,80], gamma=0.1)
        else:
            raise ValueError("no such a "+ optim_type + "optim, you can try Adam or SGD.")

    def set_transform(self, mode):
        transform = []
        if mode == 'train':
            transform.append(transforms.RandomHorizontalFlip())
            transform.append(transforms.RandomRotation(degrees=30))  # 旋转30度
            # transform.append(RandomBrightness())
            # transform.append(RandomContrast())
            # transform.append(RandomHue())
            # transform.append(RandomSaturation())
        # the advising transforms way in imagenet
        # the input image should be resized as 224 * 224 for resnet.
        transform.append(transforms.ToPILImage())
        transform.append(transforms.Resize(size=(224, 224))) # test no resize operation.
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5]))
        
        transform = transforms.Compose(transform)
        self.transform = transform


    # self define loss function
    def BCE_loss(self, input_, target):
        # cost_matrix = [1 for i in range(len(self.selected_attrs))]
        loss = F.binary_cross_entropy(input_.to(self.device),  
                                    target.type(torch.FloatTensor).to(self.device), 
                                    weight=self.attr_loss_weight.type(torch.FloatTensor).to(self.device))
        return loss

    # def focal_loss(self, inputs, targets):
    #     focal_loss_func = FocalLoss()
    #     focal_loss_func.to(self.device)
    #     return focal_loss_func(inputs, targets)

    def load_model_dict(self, model_state_dict_path):
        self.model_save_path = model_state_dict_path
        self.model.load_state_dict(torch.load(model_state_dict_path))
        print("The model has loaded !")

    def save_model_dict(self, model_state_dict_path):
        torch.save(self.model.state_dict(), model_state_dict_path)
        print("The model has saved!")
        
    def predict(self, image):
        self.model.eval()
        with torch.no_grad():
            image = torch.stack([self.transform(x) for x in image]).to(self.device)
            output = self.model(image)
            return output
