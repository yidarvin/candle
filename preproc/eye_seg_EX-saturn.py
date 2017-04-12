import h5py
import numpy as np
from os import listdir,mkdir
from os.path import isdir,join
import scipy
import scipy.misc
from skimage.filters import threshold_otsu
from skimage import measure,exposure

# Variables
size = 512

# Filepaths
path_root = '/media/dnr/Documents/data/eye_seg'
path_save = '/media/dnr/Documents/data/eye_seg_EX'
if not isdir(path_save):
    mkdir(path_save)
path_save = join(path_save, 'training')
# Filepath Corollaries
path_ex   = join(path_root, 'e_optha_EX')
path_ex_anns = join(path_ex, 'Annotation_EX')
path_ex_imgs = join(path_ex, 'EX')
path_ex_healthy = join(path_ex, 'healthy')
path_ma   = join(path_root, 'e_optha_MA')
path_ma_anns = join(path_ma, 'Annotation_MA')
path_ma_imgs = join(path_ex, 'MA')
path_ma_healthy = join(path_ex, 'healthy')
if not isdir(path_save):
    mkdir(path_save)

# Quick function for cropping and rescaling images.
def tight_crop(img,ann,size=None):
    img_gray = np.mean(img, 2)
    img_bw = img_gray > threshold_otsu(img_gray)
    img_label = measure.label(img_bw, background=0)
    largest_label = np.argmax(np.bincount(img_label.flatten())[1:]) + 1
    img_circ = (img_label == largest_label)
    img_xs = np.sum(img_circ, 0)
    img_ys = np.sum(img_circ, 1)
    xs = np.where(img_xs > 0)
    ys = np.where(img_ys > 0)
    y_lo = np.min(ys)
    y_hi = np.max(ys)
    x_lo = np.min(xs)
    x_hi = np.max(xs)
    img_crop = img[y_lo:y_hi, x_lo:x_hi, :]
    ann_crop = ann[y_lo:y_hi, x_lo:x_hi]
    if size:
        img_crop = scipy.misc.imresize(img_crop,[size,size])
        ann_crop = scipy.misc.imresize(ann_crop,[size,size],interp='nearest')
    img_crop = img_crop.astype(np.float32)
    img_crop /= 255
    for i in range(3):
        img_crop[:,:,i] = exposure.equalize_adapthist(img_crop[:,:,i], clip_limit=0.03)
    ann_crop = (ann_crop > 0) + 0
    ann_crop = scipy.ndimage.morphology.binary_dilation(ann_crop, iterations=size/56)
    ann_crop = ann_crop.astype(np.int64)
    return img_crop,ann_crop

# Crawl through EXs
list_pats = listdir(path_ex_imgs)
for name_pat in list_pats:
    path_imgs = join(path_ex_imgs, name_pat)
    path_anns = join(path_ex_anns, name_pat)
    list_imgs = listdir(path_imgs)
    list_anns = listdir(path_anns)
    for i,name_img in enumerate(list_imgs):
        #finding correlating images/annotations
        name_img_split = name_img.split('.')
        if name_img_split[-1] not in ['jpg', 'png', 'JPG']:
            print name_img_split[-1]
            continue
        if name_img[0] == '.':
            continue
        for name_ann in list_anns:
            if name_ann[:len(name_img_split[0])] == name_img_split[0]:
                break
        #reading in the images
        path_img = join(path_imgs, name_img)
        path_ann = join(path_anns, name_ann)
        img = scipy.misc.imread(path_img)
        ann = scipy.misc.imread(path_ann)
        #rescale
        path_save_pat = join(path_save, name_pat)
        if not isdir(path_save_pat):
            mkdir(path_save_pat)
        path_h5 = join(path_save_pat, name_img_split[0] + '.h5')
        img,ann = tight_crop(img,ann,size=size)
        h5f = h5py.File(path_h5)
        h5f.create_dataset('data', data=img)
        h5f.create_dataset('seg', data=ann)
        h5f.create_dataset('name', data=name_img)
        h5f.create_dataset('pat', data=name_pat)
        h5f.close()
