import torch.cuda as cutorch
import sys
import os
import argparse
import numpy as np
from glob import glob
from os.path import join
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from matplotlib.pyplot import imshow
from eval import eval_net
from unet import UNet
import random
import cv2
from utils import to_categorical, batch, split_train_val, load_resize, get_imgs_and_masks



def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5,
              dir_img=None,
              dir_mask=None,
              dir_checkpoint = None,
              channels = 1,
              classes = 1):

    ids = os.listdir(dir_img)

    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint, mode=0o755)

    iddataset = split_train_val(ids, val_percent)

    print('Starting training:')
    print('Epochs: ' + str(epochs))
    print('Batch size: ' + str(batch_size))
    print('Learning rate: ' + str(lr))
    print('Training size: ' + str(len(iddataset['train'])))
    print('Validation size: ' + str(len(iddataset['val'])))
    print('Checkpoints: ' + str(save_cp))

    N_train = len(iddataset['train'])

    optimizer = optim.RMSprop(net.parameters(),lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

        epoch_loss = 0

        # Run Batch
        for i, b in enumerate(batch(train, batch_size)):

            # Grab data
            try:
                imgs = np.array([i[0] for i in b]).astype(np.float32)
                true_masks = np.array([i[1] for i in b])
            except:
                print('prob have dimension issues, wrong orientations or half reconned images')
            # Deal with dimension issues
            if channels == 1:
                imgs = np.expand_dims(imgs,1)
            if classes>1:
                true_masks = to_categorical(true_masks,num_classes=classes)

            # Play in torch's sandbox
            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            # Send to GPU
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            # Predicted segmentations
            masks_pred = net(imgs)

            # Flatten
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat = true_masks.view(-1)

            # Calculate losses btwn true/predicted
            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            # Batch Loss
            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Epoch Loss
        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if 1:
            val_dice = eval_net(net, val, epoch,dir_checkpoint,gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       os.path.join(dir_checkpoint, 'CP{}.pth'.format(epoch + 1)))
            print('Checkpoint {} saved !'.format(epoch + 1))



def get_args(debug=False, args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img', metavar='img',
                      default=None, help='location of images')
    parser.add_argument('-m', '--mask', metavar='mask',
                      default=None, help='location of masks')
    parser.add_argument('-k', '--checkpoint', metavar='checkpoint',
                      default=None, help='location to save checkpoints')
    parser.add_argument('--epochs', '-e', metavar='epochs', default=10, type=int,
                      help='number of epochs')
    parser.add_argument('-b', '--batch-size', metavar='batchsize', default=10,
                      type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', metavar='lr', default=0.05,
                      type=float, help='learning rate')
    parser.add_argument('-g', '--gpu', metavar='gpu',
                      default=True, help='use cuda')
    parser.add_argument('-c', '--load', metavar='load',
                      default=False, help='load file model')
    parser.add_argument('-s', '--scale', metavar='scale', type=float,
                      default=1, help='downscaling factor between 0 and 1 of the images')
    parser.add_argument('-n', '--gpunum', metavar='gpunum', type=int,
                        default=0, help='which gpu to use')

    if debug==False:
        return parser.parse_args()
    else:
        return parser.parse_args(args)

def main(args):
    # If there is more than one channel, adapt unet for the channel depth
    try:
        n_channels = np.load(glob.glob(os.path.join(args.img, '*.npy'))[0]).shape[2]
    except:
        n_channels = 1


    # Change number of classes
    max0 = np.max(np.load(glob(os.path.join(args.mask, '*.npy'))[0]))
    max1 = np.max(np.load(glob(os.path.join(args.mask, '*.npy'))[1]))
    max2 = np.max(np.load(glob(os.path.join(args.mask, '*.npy'))[2]))
    max3 = np.max(np.load(glob(os.path.join(args.mask, '*.npy'))[3]))
    max4 = np.max(np.load(glob(os.path.join(args.mask, '*.npy'))[4]))

    n_classes = int(np.max((max0,max1,max2,max3,max4))+1)

    net = UNet(n_channels=n_channels, n_classes=n_classes)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        torch.cuda.set_device(args.gpunum)
        net.cuda()
        cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.learning_rate,
                  gpu=args.gpu,
                  img_scale=args.scale,
                  dir_img = args.img,
                  dir_mask = args.mask,
                  dir_checkpoint = args.checkpoint,
                  channels = n_channels,
                  classes = n_classes
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == '__main__':
    args = get_args()
    main(args)