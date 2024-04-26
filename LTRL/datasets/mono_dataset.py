from __future__ import absolute_import, division, print_function
import glob
import json

import os
import random
import cv2
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
from PIL import ImageFile

import torch
import torch.utils.data as data
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES=True

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
    - data_path: path to the data folder containing left folders
    - height: height of input images
    - width: width of input images
    - frame_idxs: indices of frames to load (if None, all are loaded)
    - is_train: whether in training mode
    - num_scales: number of scales in the image pyramid
    """
    def __init__(self,
                 data_path,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='jpg', 
                 camera_intrinsics_pth = '', 
                 num_dec = 8, # size of number in filename
                ):
        super(MonoDataset, self).__init__()

        # camera intrinsics- either load from path or pre-set intrinsics
        if len(camera_intrinsics_pth) > 0:
            self.K = np.loadtxt(camera_intrinsics_pth)
            # TODO scale with original height and width
            #self.K[0,:] /= width
            #self.K[1,:] /= height
            self.K[0,:] /= self.K[0,2]
            self.K[1,:] /= self.K[1,2]
            # multiply by new h and w
            self.K[0,:] *= width/2
            self.K[1,:] *= height/2


        else:
            """ self.K = np.array([[0.82, 0, 0.5, 0],
                            [0, 1.02, 0.5, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float64) """
            self.K =  np.array([[0.82, 0, 0.5],
                            [0, 1.02, 0.5],
                            [0, 0, 1]], dtype=np.float64)
        self.data_path = data_path
        # files to reconstruct
        self.filenames =  sorted(glob.glob(f'{self.data_path}/*{img_ext}'))
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.LANCZOS

        # frame indeces (where 0 is current frame, 1 is next frame, -1 for previous frame)
        self.frame_idxs = frame_idxs
        self.frame_idx_min = min(frame_idxs)
        self.frame_idx_max = max(frame_idxs)

        #self.frame_idxs = frame_idxs
        #self.frame_idx_min = min(frame_idxs)
        #self.frame_idx_max = max(frame_idxs)
        self.is_train = is_train
        self.img_ext = img_ext
        self.num_dec = num_dec

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
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

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
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

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, frame_index, side="l"):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        # determine if to do data augmentation for trainings datasets
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        # checking if we can take the surrounding frames (eg if we take the first frame of a sequence, there won't be -1 frame) 
        filedir = self.filenames[frame_index]
        folderdir = os.path.dirname(filedir) # name of folder where the data is located

        # grabbing frame index from file name (eg. frame number 26)- first take name of file (basename=000XXX.jpg), then remove the extension .jpg ([:-4]) and convert to int
        frame_index = int(os.path.basename(filedir)[:-4]) 
        # checking there's a file for the max index we're grabbing for prediction. Eg if we're on the last frame, don't try to grab frame_index+1
        file_min = "{:0{}d}{}".format((frame_index+self.frame_idx_min)-1,self.num_dec,self.img_ext) # Note- even if it's -, it's still + idx the idx itself will be -ve number
        file_max = "{:0{}d}{}".format((frame_index+self.frame_idx_max)-1,self.num_dec,self.img_ext)
        # if there's not a file for the min/max index, adjust frame_index accordingly so we slide to a frame where there is the required files
        if not os.path.exists(os.path.join(folderdir, file_min)):
            frame_index += 1
        elif not os.path.exists(os.path.join(folderdir, file_max)):
            frame_index -= 1

        # apply necessary transforms to the selected frames
        for img_idx in self.frame_idxs:
            # create file name correctly with the selected idx
            filename = "{:0{}d}{}".format((frame_index+img_idx)-1,self.num_dec,self.img_ext)
            filedir = os.path.join(folderdir, filename)
            # load image of this file name
            img = Image.open(filedir).convert('RGB')
            if do_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            inputs[("color", img_idx, -1)] = img


        # COLOR AUGMENTATION
        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness,self.contrast,self.saturation,self.hue)
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

        if self.load_depth:
            depth_gt = self.get_depth(folderdir, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
   
        return inputs
    
    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(frame_index))
        
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        return color


    def check_depth(self):
        return None
    

    def get_depth(self, folder, frame_index, side, do_flip):
        print('getting depth')
        f_str = "scene_points{:0{}d}.tiff".format(frame_index-1, self.num_dec)
        sequence = folder[7]
        data_splt = "train" if int(sequence) < 8 else "test"
        # depth_path = os.path.join(
        #     self.data_path, data_splt, folder, "data", self.side_map[side] + "_depth",
        #     f_str)

        # depth_gt = cv2.imread(depth_path, 2)
        
        """
        depth_path = os.path.join(
            self.data_path, data_splt, folder, "data", "scene_points",
            f_str)
        """
        
        depth_path = os.path.join(
            data_splt, folder, "data", "scene_points",
            f_str)

        depth_gt = cv2.imread(depth_path, 3)
        depth_gt = depth_gt[:, :, 0]
        depth_gt = depth_gt[0:1024, :]
        
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
    
    
    def get_pose(self, folder, frame_index):
        f_str = "frame_data{:0{}d}.json".format(frame_index-1, self.num_dec)
        sequence = folder[7]
        data_splt = "train" if int(sequence) < 8 else "test"
        """ 
        pose_path = os.path.join(
            self.data_path, data_splt, folder, "data", "frame_data",
            f_str) """
        pose_path = os.path.join(
            data_splt, folder, "data", "frame_data",
            f_str)
        with open(pose_path, 'r') as path:
            data = json.load(path)
            pose = np.linalg.pinv(np.array(data['camera-pose']))
            # pose = np.array(data['camera-pose'])
        
        return pose

    
    def get_image_path(self, frame_index):
        image_path = self.filenames[frame_index]
        return image_path

