from shutil import copyfile
# copyfile(src,dst)
import pathlib

from skimage.io import imread_collection
vdo = '6'
imgs = imread_collection('/home/mgharasu/Videos/traffic camera/keyframes/'+vdo+'/*.jpg')#/*.jpg')#frame_number/*.jpg')
a=0
for img in imgs.files:
    tmp = img.split('/')[-1][3:]
    copyfile('/home/mgharasu/Videos/traffic camera/'+vdo+'/'+tmp,'/home/mgharasu/Videos/traffic camera/keyframes/new'+vdo+'/'+tmp)
