import datasets
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt

import networks
from layers import disp_to_depth
from datasets.mono_dataset import MonoDataset
from options import MonodepthOptions



def evaluate(opt): 
    data_path = '/Users/aure/Documents/CARES/code/mono_reconstruction/LTRL/data/d3k4-phantom'
    dataset= MonoDataset(data_path,
                 opt.height,
                 opt.width,
                 4,
                 is_train=False,
                 img_ext='jpg')
    #dataset = SCAREDRAWDataset(opt.data_path, is_train=False)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,pin_memory=False, drop_last=True)

    # get one sample and take the color image
    for inputs in dataloader:
        #inputs = next(iter(dataloader))
        sample = inputs[("color_aug", 1, 0)]

        # load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder_path = "/Users/aure/Documents/CARES/code/mono_reconstruction/LTRL/af-sfmlearner/encoder.pth"
        decoder_path = "/Users/aure/Documents/CARES/code/mono_reconstruction/LTRL/af-sfmlearner/depth.pth"
        encoder_dict = torch.load(encoder_path, map_location=device)
        encoder = networks.ResnetEncoder(18, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))

        
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path, map_location=device))


        encoder.to(device)
        encoder.eval()
        depth_decoder.to(device)
        depth_decoder.eval()
        sample = sample.to(device)

        output = depth_decoder(encoder(sample))
        pred_disp_resized, _ = disp_to_depth(output[("disp", 0)], 1e-3, 150)
        pred_disp_tensor = pred_disp_resized.cpu()[:, 0].detach()
        pred_disp = pred_disp_resized.cpu()[:, 0].detach().numpy()
        sample = sample.cpu()
        vmax = np.percentile(pred_disp, 95)


        plt.subplot(121), plt.imshow(sample[0].permute(1,2,0))
        plt.axis('OFF')
        plt.title('Input')

        plt.subplot(122), plt.imshow(pred_disp[0], cmap='magma', vmax=vmax)
        plt.axis('OFF')
        plt.title('Depth Prediction');
        plt.show()
    return 


if __name__=='__main__': 
    options = MonodepthOptions()
    evaluate(options.parse())