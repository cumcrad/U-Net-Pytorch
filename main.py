import os
import sys
import cv2
import nibabel as nib
import numpy as np
from matplotlib.pyplot import imshow
from os.path import join
from glob import glob
sys.path.append('/data/mike/src/tools')
import imgtools as imt
import miketools as mt
sys.path.append('/data/mike/src/Pytorch-UNet')
import train1
import predict

class patient:
    def __init__(self,fn,out,test=False):
        self.fn=fn
        self.imgpath = join(os.path.dirname(os.path.dirname(fn)),'img')
        self.basename = os.path.basename(fn)
        self.out = out
        pathsegs = self.basename.split('_')
        self.id = pathsegs[0]
        self.test = test
    def getimg(self,channels=1):
        for channel in range(channels):
            fn = glob(join(self.imgpath, self.id + '*' + '.nii*'))[channel]
            if channel ==0:
                self.img = nib.load(fn).get_data()
            else:
                self.img = np.stack((self.img,nib.load(fn).get_data()), axis=3)
    def get_seg(self, classes=1):
        self.seg = nib.load(self.fn).get_data()
        # Set all segs to 1
        if classes==1:
            self.seg[np.nonzero(self.seg)] = 1
    def reshape(self,shape=256):
        if self.img.shape[0] != shape or self.img.shape[1] != shape:
            scale = shape/self.img.shape[0]
            self.img = cv2.resize(self.img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            self.seg = cv2.resize(self.seg, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    def cut_unsegmented(self):
        # initialize img placeholder
        tmpimg = np.zeros(self.img.shape)
        # initialize seg placeholder
        try:
            tmpseg = np.zeros(self.seg.shape)
        except:
            print('err')
        # Iterate through z dim
        n=0
        for z in range(self.img.shape[2]):
            # if there is segmentation present
            if self.seg[:, :, z].max() > 0:
                # put img and seg in tmp area
                tmpimg[:, :, n] = self.img[:, :, z]
                tmpseg[:, :, n] = self.seg[:, :, z]
                n=n+1
        self.img = tmpimg[:, :, :n]
        self.seg = tmpseg[:, :, :n]
        self.cut = 1
    def crop(self,shape=256):
        if len(self.img.shape == 4):
            tmpimg = np.zeros((shape, shape, self.img.shape[2],self.img.shape[3]))
        elif len(self.img.shape == 3):
            tmpimg = np.zeros((shape, shape, self.img.shape[2]))
        else:
            print('cropping fails because dimensions are funny')
        tmpseg = np.zeros((shape, shape, self.img.shape[2]))
        n=0

        for i in range(self.img.shape[2]):
            [x1, x2, y1, y2] = imt.bbox(self.seg[:,:,i])
            if x1+x2+y1+y2>0:
                ymid = int((y2 + y1) / 2)
                xmid = int((x2 + x1) / 2)
                crop2 = int(shape / 2)

                # Crop a box that is centered
                x1n = int(xmid - crop2)
                x2n = int(xmid + crop2)
                y1n = int(ymid - crop2)
                y2n = int(ymid + crop2)

                # Boundary Conditions
                if x2n > self.img.shape[0]:
                    (x1n, x2n) = (int(self.img.shape[0] - shape), int(self.img.shape[0]))
                elif x1n < 0:
                    (x1n, x2n) = (int(0), int(self.img.shape[0] / 2))
                if y2n > self.img.shape[1]:
                    (y1n, y2n) = (int(self.img.shape[1] - shape), int(self.img.shape[1]))
                elif y1n < 0:
                    (y1n, y2n) = (int(0), int(self.img.shape[1] / 2))

            else:
                crop2 = int(shape / 2)
                x1n = int(self.img.shape[0] / 2 - crop2)
                x2n = int(self.img.shape[0] / 2 + crop2)
                y1n = int(self.img.shape[1] / 2 - crop2)
                y2n = int(self.img.shape[2] / 2 + crop2)

            tmpimg[:, :, i] = self.img[y1n:y2n, x1n:x2n, i]
            tmpseg[:, :, i] = self.seg[y1n:y2n, x1n:x2n, i]

        self.img = tmpimg
        self.seg = tmpseg
        self.cropped = 1
    def exportnpys(self):
        for z in range(self.img.shape[2]):
            fn = self.id + '_' + str(z).zfill(3) + '.npy'
            if self.test ==False:
                imgfn = join(self.out, 'img', fn)
                segfn = join(self.out, 'seg', fn)
            if self.test == True:
                imgfn = join(self.out, 'tstimg', fn)
                segfn = join(self.out, 'tstseg', fn)
            np.save(imgfn,self.img[:, :, z])
            np.save(segfn,self.seg[:, :, z])
def mkdir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path, mode=0o755)
    except:
        print('could not make dir ' + path)
def dice(staticfn,movefn, k=1):
    # k is the # of the segmentation if there are more than one class
    static = np.load(staticfn)
    move = np.load(movefn)
    return np.sum(static[move==k])*2.0 / (np.sum(static) + np.sum(move))
def dcm2nii(file):
    executableloc = '/home/exx/programs/mricron/dcm2nii -4 N '
    command = executableloc + file
    os.system(command)
def loadnii(segniis, test=0.2, shape=256,crop=256,channels=1,classes=1):
    # Put aside test set
    np.random.shuffle(segniis)
    transition = int(test*len(segniis))
    testniis=segniis[:transition]
    trainniis = segniis[transition:]

    # Create Training dataset
    for seg in trainniis:
        # load all data to patient class
        patobj = patient(seg, out)
        patobj.getimg(channels)
        patobj.get_seg(classes)
        patobj.reshape(shape)
        patobj.cut_unsegmented()
        if crop !=256:
            patobj.crop(crop)
        patobj.exportnpys()

    # Create Testing output
    for seg in testniis:
        # load all data to patient class
        patobj = patient(seg, out, test=True)
        patobj.getimg(channels)
        patobj.get_seg(classes)
        patobj.reshape(shape)
        patobj.cut_unsegmented()
        if crop != 256:
            patobj.crop(crop)
        patobj.exportnpys()

# Define variables
indir   = '/data/mike/breast/neo/data_1_n4'
out     = '/data/mike/breast/neo/unet'

# Prep - DCM2NII

# Load and Preprocess NIIs args
segniis = glob(join(indir,'seg','*.nii.gz'))
shape=256
crop=256
channels=2
classes=1

# Train Args
epochs = '50'
lr = '0.000005'
scale = '1'
load = False
gpunum = 0

# Predict Args
model= os.path.join(out,'CP100.pth')
cpu = 'False'
no_crf = 'True'
mask_threshold = 0.5

# Create output folders
inimg   = join(indir,'img')
inseg   = join(indir,'seg')
img     = join(out,'img')
seg     = join(out,'seg')
pred    = join(out,'segpred')
results = join(out,'results.npy')
mkdir(out),mkdir(img),mkdir(seg)

# Load
# loadnii(segniis,shape,crop,channels,classes)

# Train
args = train1.get_args(debug=True, args=['-i', img, '-m', seg, '-k', out, '-l' ,lr , '-c', load , '-e', epochs, '-s', scale, '-n', gpunum])
train1.main(args)

# Predict
in_fn_list = glob(os.path.join(img,'*.npy'))
out_fn_list = in_fn_list.copy()
for i,in_fn in enumerate(in_fn_list):
    out_fn_list[i]=os.path.join(pred,os.path.basename(in_fn))
predargs = predict.get_args(debug=True, args=['-m', model, '-i', in_fn_list, '-o', out_fn_list,  '-s',scale])
predict.main(predargs)

# Results

# Plots