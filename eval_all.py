import os
import torch
from tqdm import tqdm
from LTRL.datasets.mono_dataset import MonoDataset
from torch.utils.data import DataLoader
import LTRL.networks as networks
from LTRL.layers import transformation_from_parameters, disp_to_depth

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def evaluate_pose():
    return

def evaluate_depth():
    return


def reconstruct_pointcloud(rgb, depth, cam_K, vis_rgbd=False):

    rgb = np.asarray(rgb, order="C")
    rgb_im = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_im = o3d.geometry.Image(depth)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im, convert_rgb_to_intensity=False)
    if vis_rgbd:
        plt.subplot(1, 2, 1)
        plt.title('RGB image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Depth image')
        plt.imshow(rgbd_image.depth)
        plt.colorbar()
        plt.show()

    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = cam_K
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        cam
    )

    return pcd

def evaluate_all(
        data_path = '/Users/aure/Documents/CARES/code/mono_reconstruction/data/rec_aug/august_recordings/rec_10_endo', # path where data to be evaluated is stored
        img_ext = '.png',
        load_weights_folder="LTRL/af-sfmlearner", #name of model to load   
        height=256, width=320, #input image height and width
        intrinsics_pth = '/Users/aure/Documents/CARES/code/mono_reconstruction/data/rec_aug/august_recordings/zoomed_calibration/intrinsics_endo.txt',
        num_dec = 8, # size of number in filename

        # eval
        batch_size = 1, # batch size for evaluation
        num_workers = 1, # number of workers for dataloader
        num_layers = 18 ,# "number of resnet layers", choices=[18, 34, 50, 101, 152]
        num_scales = 4, # number of scales to evaluate
        frame_idxs = [0,1], # frame indices to evaluate on (current and one after)
        # save
        save_dir = 'results/aug_10', # directory to save the results

        # depth scaling
        min_depth = 1e-3,
        max_depth = 150,

        visualise_depth = False,


):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # ------------------------------ loading data -------------------------------------------
    print(f"-> Loading data from {data_path}")
    dataset = MonoDataset(data_path,
                 height,
                 width,
                 frame_idxs,
                 num_scales, #num scales
                 is_train=False,
                 img_ext=img_ext,
                 camera_intrinsics_pth=intrinsics_pth, 
                 num_dec= num_dec,
                 )
    
    
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=False,
                            pin_memory=True, 
                            drop_last=False,  ###### TO ASK- WHAT IS THIS
                            num_workers=num_workers)
    

    # ------------------------------ loading depth model with weights -------------------------------------------
    print(f'initialising depth model')
    # initialise depth model RESNET( num_layers, pretrained, num_input_images=1)
    depth_encoder = networks.ResnetEncoder(num_layers, pretrained=False, num_input_images=1) 
    depth_decoder = networks.DepthDecoder(depth_encoder.num_ch_enc, scales=range(num_scales))
    # weights paths
    encoder_weights_path = os.path.join(load_weights_folder, "encoder.pth")
    decoder_weights_path = os.path.join(load_weights_folder, "depth.pth")
    depth_encoder_dict = torch.load(encoder_weights_path, map_location=DEVICE)
    depth_decoder_dict = torch.load(decoder_weights_path, map_location=torch.device(DEVICE))

    # load weights to depth model
    print(f"-> Loading weights from {load_weights_folder} encoder.pth and depth.pth")
    model_dict = depth_encoder.state_dict()
    depth_encoder.load_state_dict({k: v for k, v in depth_encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(depth_decoder_dict)

    # setting models in evaluation mode
    depth_encoder.to(DEVICE)
    depth_decoder.to(DEVICE)
    depth_encoder.eval()
    depth_decoder.eval()

    # ------------------------------ loading pose model with weights -------------------------------------------
    print(f"Initialising pose model")
    # initialise pose model
    pose_encoder = networks.ResnetEncoder(num_layers, pretrained=False, num_input_images=len(frame_idxs))
    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=len(frame_idxs), stride=1)

    # weights paths
    print(f'-> Loading weights from {load_weights_folder} pose_encoder.pth and pose.pth')
    encoder_weights_path = os.path.join(load_weights_folder, "pose_encoder.pth")
    decoder_weights_path = os.path.join(load_weights_folder, "pose.pth")
    pose_encoder_dict = torch.load(encoder_weights_path, map_location=torch.device(DEVICE))
    pose_decoder_dict = torch.load(decoder_weights_path, map_location=torch.device(DEVICE))

    # load weights to model
    pose_encoder.load_state_dict(pose_encoder_dict)
    pose_decoder.load_state_dict(pose_decoder_dict)

    # setting models in evaluation mode
    pose_encoder.to(DEVICE)
    pose_decoder.to(DEVICE)
    pose_encoder.eval()
    pose_decoder.eval()    
    
    # ------------------------------ evaluating model -------------------------------------------
    pred_poses = []
    pred_depth = []
    predicted_pc = []
    with torch.no_grad(): # TODO ask why if we set .eval()
        for i, batch in tqdm(enumerate(dataloader)):
            # get the input data
            inputs = batch
            # move input data to device (cpu most likely)
            if DEVICE != 'cpu':
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(DEVICE)

            # get the color images
            # TODO ask why 1 then 0 and not other way round- should I loop to get all frame_idx?
            all_color_aug = torch.cat([inputs[("color", 0, 0)], inputs[("color", 1, 0)]], 1)
            input_data = all_color_aug.to(DEVICE)

            # ---------------------------- PREDICT POSE
            # predict pose- first pass through the encoder to get the features and then the pose decoder to get the rotation and translation
            features = [pose_encoder(input_data)]
            axisangle, translation = pose_decoder(features)
            predicted_pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy()

            # ------------------------------ PREDICT DEPTH
            # predict depth- first pass through the encoder to get the features and then the depth decoder to get the depth
            predicted_disp = depth_decoder(depth_encoder(input_data[:,0:3,:,:]))
            # grab unscaled depth 
            # dict_keys([('disp', 3), ('disp', 2), ('disp', 1), ('disp', 0)])
            predicted_disp_original, _ = disp_to_depth(predicted_disp[("disp", 0)], min_depth, max_depth)
            pred_depth_scale_1, _ = disp_to_depth(predicted_disp[("disp", 1)], min_depth, max_depth)

            # squeeze channel- convert to np
            predicted_disp = predicted_disp_original.cpu()[:, 0].detach().numpy()
            predicted_depth = (1/predicted_disp).squeeze()
            predicted_depth[predicted_depth < min_depth] = min_depth
            predicted_depth[predicted_depth > max_depth] = max_depth
            vmax = np.percentile(predicted_depth, 95)

            # --------------------- results
            if visualise_depth:
                plt.subplot(121), plt.imshow(input_data[0,0:3,:,:].permute(1,2,0))
                plt.axis('OFF')
                plt.title('Input')

                plt.subplot(122), plt.imshow(predicted_depth[0], cmap='magma', vmax=vmax)
                plt.axis('OFF')
                plt.title('Depth Prediction');
                pred_depth.append(predicted_depth)

            # reconstruct pointcloud
            rgb = input_data[0,0:3,:,:].squeeze().permute(1,2,0).cpu().numpy() * 255
            pcd = reconstruct_pointcloud(rgb, predicted_depth, dataset.K, vis_rgbd=False)

            # save the results
            fn = os.path.join(save_dir, f'{i}.ply')
            o3d.io.write_point_cloud(fn, pcd)
            predicted_pc.append(pcd)
            # save pose in txt file
            pred_poses.append(predicted_pose)
            #np.savetxt(os.path.join(save_dir, f'{i}_pose.txt'), predicted_pose.squeeze())
    
    # save results
    print(f'saving poses and depths in {save_dir}')
    np.save(os.path.join(save_dir, 'pred_poses.npy'), np.array(pred_poses))
    np.save(os.path.join(save_dir, 'pred_depth.npy'), np.array(pred_depth))
    return

def main(): 
    evaluate_all(
        data_path = '/Users/aure/Documents/CARES/code/mono_reconstruction/data/rec_aug/august_recordings/rec_10_endo', # path where data to be evaluated is stored
        load_weights_folder="LTRL/af-sfmlearner", #name of model to load   
        height=256, width=320, #input image height and width,
        intrinsics_pth = '/Users/aure/Documents/CARES/code/mono_reconstruction/data/rec_aug/august_recordings/zoomed_calibration/intrinsics_endo.txt',
        #intrinsics_pth = '',
        img_ext = '.png',
        num_dec = 8, # size of number in filename

        # eval
        batch_size = 1, # batch size for evaluation
        num_workers = 1, # number of workers for dataloader
        num_layers = 18 ,# number of resnet layers", choices=[18, 34, 50, 101, 152]
        num_scales = 4, # number of scales to evaluate

        # save
        save_dir = 'results/aug_10_scaling',
    )

    return 


if __name__=='__main__': 
    # determine the device to be used for training and evaluation
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    main() 