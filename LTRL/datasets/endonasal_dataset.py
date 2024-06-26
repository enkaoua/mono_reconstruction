from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
from PIL import ImageFile

import torch
import torch.utils.data as data
from torchvision import transforms
import skimage.transform
import PIL.Image as pil
import cv2
from glob import glob

ImageFile.LOAD_TRUNCATED_IMAGES=True

class EndonasalDataset(data.Dataset):
    """Endonasal dataloader
    """
    def __init__(self,
                 root,
                 seqs,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False, 
                 img_ext='.png', 
                 mask_path='', 
                 camera_intrinsics_pth='', 
                 height_intrinsics=False):
        super(EndonasalDataset, self).__init__()
        
        self.filenames = []
        for seq in seqs:
            frame_path = glob (root +'/'+ seq + f'/*{img_ext}')
            frame_path.sort()
            if len(frame_idxs)>1:
                frame_path = frame_path[1:-1]
            self.filenames.extend(frame_path)
        
        self.mask_path = mask_path
        
        if len(self.mask_path)>0:
            """ # load png mask image
            mask_img = cv2.imread(self.mask_path)
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            # convert img to mask
            self.mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)[1] """
            mask_img = cv2.imread(self.mask_path)
            # convert to grayscale and generate mask
            gray = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
            mask = np.zeros(gray.shape)
            gray = np.array(gray, np.uint8)
            # detect largest circle
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 3, 100)
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                (x, y, r) = circles[0:1][0]
                print('circle found:',x, y, r-80)
                cv2.circle(mask, (x, y), r-30, 255, -1)
            self.mask = mask

        # resize mask
        self.resized_mask = cv2.resize(self.mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.LANCZOS
        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.to_tensor = transforms.ToTensor()

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.transforms.ColorJitter(self.brightness,self.contrast,self.saturation,self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
        self.load_depth = self.check_depth()
        
        # camera intrinsics- either load from path or pre-set intrinsics
        if len(camera_intrinsics_pth) > 0:
            intrinsics = np.loadtxt(camera_intrinsics_pth, dtype=np.float32)
            # normalising intrinsics
            #self.K[0,:] /= 2*self.K[0,2]
            #self.K[1,:] /= 2*self.K[1,2]
            # multiply by new h and w
            
            if height_intrinsics:
                intrinsics[0,:] /= 1920
                intrinsics[1,:] /= 1080
            else:
                intrinsics[0,:] *= width/2
                intrinsics[1,:] *= height/2
            
            self.K = np.eye(4, dtype=np.float32)
            self.K [0:3,0:3] = intrinsics

            print("--------> INTRINSICS", self.K)


            

        else:
            self.K = np.array([[0.76, 0, 0.47, 0],
                           [0, 1.35, 0.45, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        
        filedir = self.filenames[index]
        inputs['filepath'] = filedir
        folderdir = os.path.dirname(filedir)
        name_digits = os.path.basename(filedir)[:-4]
        frame_index = int(name_digits)
        img_ext, name_length = filedir[-4:], len(name_digits)
        
        for i in self.frame_idxs:
            filename = "{:0>{width}}".format(frame_index + i, width=name_length)
            try:
                img = Image.open(os.path.join(folderdir, filename+img_ext)).convert('RGB')
                if len(self.mask_path)>0:
                
                    # resize mask to size of image
                    #mask = cv2.resize(mask, (self.opt.height, self.opt.width))
                    # make size of mask repeated to match batch and RGB (size 12, 3, x,y)

                    #mask = np.tile(mask, (self.opt.batch_size, 3, 1, 1)).shape
                    # convert img to np array
                    img = np.array(img)
                    # all areas outside mask should be set to 0 in predicted image
                    img[self.mask==0]=0
                    # convert back to PIL image
                    img = Image.fromarray(img)
            except:
                print(f'dataloader file name: {os.path.join(folderdir, filename+img_ext)}')
            if do_flip:
                img = img.transpose(pil.FLIP_LEFT_RIGHT)
            
            
            
            inputs[("color", i, -1)] = img
            
        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            inv_K = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        """         
        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
        """
        return inputs
    
    def check_depth(self):
        return False

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "scene_points{:06d}.tiff".format(frame_index-1)
        sequence = folder[7]
        data_splt = "train" if int(sequence) < 8 else "test"

        depth_path = os.path.join(
            data_splt, folder, "data", "scene_points",
            f_str)

        depth_gt = cv2.imread(depth_path, 3)
        depth_gt = depth_gt[:, :, 0]
        depth_gt = depth_gt[0:1024, :]

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
