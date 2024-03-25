from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import time

import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib

from layers import disp_to_depth
from utils import readlines, compute_errors
from options import MonodepthOptions
import datasets
import networks
from datasets.mono_dataset import MonoDataset


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return Image.fromarray(colors)


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(data_path='/Users/aure/Documents/CARES/code/mono_reconstruction/LTRL/data/rec_aug/august_recordings/rec_10_endo', # path where data to be evaluated is stored
            img_ext='png', # image extension
            load_weights_folder="LTRL/af-sfmlearner", #name of model to load
            height=256, width=320, #input image height and width
            eval_mono=1, eval_stereo=0, 
             ext_disp_to_eval=None, #"optional path to a .npy disparities file to evaluate"
             device = torch.device('cpu'), 
             num_layers = 18 ,# "number of resnet layers", choices=[18, 34, 50, 101, 152]
             num_workers = 12 ,#"number of dataloader workers
             post_process=True, #if set will perform the flipping post processing
             min_depth=0.1, max_depth=150, #min and max depth values to evaluate
             save_pred_disps=True,
             results_pth = 'LTRL/results/aug_10',
             
             
             # evaluation
             no_eval = True, #if set will not evaluate the model
             eval_split = "eigen", #choices=["test", "eigen", "benchmark"]
             disable_median_scaling = True, #if set will not scale the results by the median of the ground truth depth
             pred_depth_scale_factor = 1, #if set will scale the results by this factor
             visualize_depth = True
             ):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150

    assert sum((eval_mono, eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if ext_disp_to_eval is None:

        # read model with weigths
        load_weights_folder = os.path.expanduser(load_weights_folder)
        assert os.path.isdir(load_weights_folder), \
            "Cannot find a folder at {}".format(load_weights_folder)
        print("-> Loading weights from {}".format(load_weights_folder))

        #filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path, map_location=device)

        """ dataset = datasets.SCAREDRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False) """
        dataset = MonoDataset(data_path,
                 height,
                 width,
                 4,
                 is_train=False,
                 img_ext=img_ext)
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=num_workers,
                                pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path, map_location=device))

        if torch.cuda.is_available():
            encoder.cuda()
            depth_decoder.cuda()

        encoder.eval()
        depth_decoder.eval()

        pred_disps = []
        inference_times = []
        #sequences = []
        #keyframes = []
        #frame_ids = []
        
        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                if torch.cuda.is_available():
                    input_color = data[("color", 1, 0)].cuda()
                else:
                    input_color = data[("color", 1, 0)]

                if post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                time_start = time.time()
                output = depth_decoder(encoder(input_color))
                inference_time = time.time() - time_start
                
                pred_disp, _ = disp_to_depth(output[("disp", 0)], min_depth, max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                inference_times.append(inference_time)
                #sequences.append(data['sequence'])
                #keyframes.append(data['keyframe'])
                #frame_ids.append(data['frame_id'])

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if save_pred_disps:
        output_path = os.path.join(
            results_pth, 'disparities.npy')
        print("-> Saving predicted disparities to ", output_path)
        if not os.path.exists(results_pth):
            os.makedirs(results_pth)
        np.save(output_path, pred_disps)

    if no_eval:
        print("-> Evaluation disabled. Done.")
        quit()
    elif eval_split == 'benchmark':
        save_dir = os.path.join(results_pth, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    

    if visualize_depth:
        vis_dir = os.path.join(results_pth, "vis_depth")
        os.makedirs(vis_dir, exist_ok=True)
        
    print("-> Evaluating")

    if eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        disable_median_scaling = True
        pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    if not no_eval:
        gt_path = os.path.join(splits_dir, eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

        errors = []
        ratios = []

        for i in range(pred_disps.shape[0]):

            #sequence = str(np.array(sequences[i][0]))
            #keyframe = str(np.array(keyframes[i][0]))
            #frame_id = "{:06d}".format(frame_ids[i][0])
            
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1/pred_disp

            if visualize_depth:
                vis_pred_depth = render_depth(pred_depth)
                #vis_file_name = os.path.join(vis_dir, sequence + "_" +  keyframe + "_" + frame_id + ".png")
                vis_file_name = os.path.join(vis_dir,  ".png")
                vis_pred_depth.save(vis_file_name)
                
            if eval_split == "eigen":
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            else:
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= pred_depth_scale_factor
            # print(pred_depth.max(), pred_depth.min())
            if not disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            
            errors.append(compute_errors(gt_depth, pred_depth))
            

        if not disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

        mean_errors = np.array(errors).mean(0)

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("average inference time: {:0.1f} ms".format(np.mean(np.array(inference_times))*1000))

        print("\n-> Done!")


def evaluate_params(opt):
    evaluate(

    )


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate_params(options.parse())
