import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from unet import UNet

import matplotlib.pyplot as plt

def get_args(debug=False, args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--vizsave', '-d', help="Save Visualizations",
                        default=False)
    parser.add_argument('--save', '-a', help="Save the output masks",
                        default=True)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

def plot_img_and_mask(img, mask, save=False, show = False, fn=None):
    fig = plt.figure()

    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    plt.imshow(img)

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    plt.imshow(mask)

    if show == True:
        plt.show()

    if save==True:
        plt.savefig(fn)

    plt.close(fig)

def predict_img(net,
                img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_gpu=False):
    net.eval()

    img = img.unsqueeze(0)

    if use_gpu:
        img = img.cuda()

    with torch.no_grad():
        output= net(img)
        probs = output.squeeze(0)
        mask_np = probs.squeeze().cpu().numpy()

    return mask_np > out_threshold


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def main(args):
    in_files = args.input
    out_files = get_output_filenames(args)

    n_channels = np.load(in_files[0]).shape[2]
    ## NPY 1 channel Uint16 medical images
    net = UNet(n_channels=n_channels, n_classes=2)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))

        img = np.load(fn)

        mask = predict_img(net=net,
                           img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_gpu=not args.cpu)

        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))
            ## save this plt
            fn = out_files[i][:-3] + 'jpg'

            if args.vizsave:
                save = True

            plot_img_and_mask(img, mask, save=save, fn=fn)

        if args.save:
            # Save NPY file
            out_fn = out_files[i]

            np.save(out_fn, mask)

            print("Mask saved to {}".format(out_files[i]))

if __name__ == "__main__":
    args = get_args()
    main(args)


