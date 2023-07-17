from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
from os import listdir
from os.path import join
import numpy as np
import torch
import torchvision.transforms as tt
import torchvision.transforms.functional as TF

class CTtoPETDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        #self.mode = opt.modeTT  
        #self.preprocess_gamma = opt.preprocess_gamma
        BaseDataset.__init__(self, opt)

        # if self.mode=='test':
        #     self.CT_dir = join(opt.dataroot, 'temp_folder')
        #     self.PET_dir = join(opt.dataroot, 'temp_folder')
        # else:
        self.opt = opt # I added
        if opt.isTrain:
            self.CT_dir = join(opt.dataroot, 'trainA')
            self.PET_dir = join(opt.dataroot, 'trainB')
        else:
            self.CT_dir = join(opt.dataroot, 'temp_folder')
            self.PET_dir = join(opt.dataroot, 'temp_folder')
        self.ids = [file for file in listdir(self.CT_dir) if not file.startswith('.') and file.endswith('.npy')]
    ####################################################################################################
    @classmethod
    def preprocessCT(cls, im, minn=-900.0, maxx=200.0, noise_std = 0):
        img_np = np.array(im)   #(5,512,512)
        # Adding Noise
        if noise_std:
            s0,s1,s2 = img_np.shape
            img_np = img_np + noise_std*np.random.randn(s0,s1,s2)
        img_np = np.clip(img_np,minn ,maxx)
        img_np = (img_np - minn)/(maxx-minn)
        return img_np
    ####################################################################################################
    # Gamma Function on PET
    @classmethod
    def preprocessPET_gamma(cls, img, gamma = 1/2, maxx = 7, noise_std = 0 ):
        img = np.array(img)
        img = img/100.0
        if noise_std:
            s0,s1,s2 = img.shape
            img = img + noise_std*np.random.randn(s0,s1,s2)
        img = np.clip(img, 0, maxx)
        img = img/maxx
        img = np.power(img, gamma)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        return img
    ####################################################################################################
    # Data Augmentation
    def transform(self, CT, PET): #(1,512,512)
        # Affine
        if torch.rand(1) < 0.95:
            #affine_params = tt.RandomAffine(0).get_params((-15, 15), (0.07, 0.07), (0.85, 1.15), (-10, 10),img_size=(512,512))
            affine_params = tt.RandomAffine(0).get_params((-45, 45), (0.10, 0.10), (0.85, 1.15), (-7 , 7 ),img_size=(512,512))
        else:
            affine_params = tt.RandomAffine(0).get_params((-180, +180), (0.10, 0.10), (0.85, 1.15), (-7 , 7 ),img_size=(512,512))
        #affine_params = tt.RandomAffine(0).get_params((-45, 45), (0.10, 0.10), (0.85, 1.15), (-7 , 7 ),img_size=(512,512))
        CT  = TF.affine(CT, *affine_params)
        PET = TF.affine(PET, *affine_params)
        return CT, PET
    ####################################################################################################
    @classmethod
    def edge_zero(cls, img):
        img[:,0,:] = 0
        img[:,-1,:] = 0
        img[:,:,0] = 0
        img[:,:,-1] = 0
        return img
    ####################################################################################################
    def __getitem__(self, i):
        # if self.mode == 'test':
        #     self.ids = np.sort(self.ids)
        idx = self.ids[i]
        PET_file = join( self.PET_dir , idx )
        CT_file = join(self.CT_dir , idx )

        # Loading
        if self.opt.isTrain == True:
            PET = np.load(PET_file)[2:5,:,:]
        else:
            PET = np.load(CT_file)[2:5,:,:]
        CT = np.load(CT_file)[2:5,:,:]
        CT = self.preprocessCT(CT)
        PET = self.preprocessPET_gamma(PET) 

        CT = self.edge_zero(CT)
        PET = self.edge_zero(PET)
        # Data augmentation
        # if self.mode == 'train':
        #     CT, PET = self.transform(  torch.from_numpy(CT), torch.from_numpy(PET)  )
        #     CT, PET = CT.type(torch.FloatTensor), PET.type(torch.FloatTensor)
        # else:
            # To float before GaussianTorch(PET)
        CT = torch.from_numpy(CT).type(torch.FloatTensor)
        PET = torch.from_numpy(PET).type(torch.FloatTensor)
        return {'A': CT, 'B': PET, 'A_paths': self.CT_dir, 'B_paths': self.PET_dir, 'name':idx}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.ids)
