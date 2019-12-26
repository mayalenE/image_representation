import numpy as np
import random
from torchvision import transforms


''' ---------------------------------------------
               PREPROCESS DATA HELPER
-------------------------------------------------'''
class Crop_Preprocess(object):
    def __init__(self, crop_type ='center', crop_ratio = 1):
        if crop_type not in ['center', 'random']:
            raise ValueError('Unknown crop type {!r}'.format(crop_type))
        self.crop_type = crop_type
        self.crop_ratio = crop_ratio
        
    def __call__(self, x): 
        if self.crop_type == 'center':
            return centroid_crop_preprocess(x, self.crop_ratio)
        elif self.crop_type == 'random':
            return centroid_crop_preprocess(x, self.crop_ratio)

def centroid_crop_preprocess(x, ratio = 2):
    '''
    arg: x, tensor 1xHxW 
    '''    
    if ratio==1:
        return x
    
    img_size = (x.size(1), x.size(2))
    patch_size = (img_size[0]/ratio, img_size[1]/ratio)  
    
    # crop around center of mass (mY and mX describe the position of the centroid of the image)
    image = x[0].numpy()
    x_grid, y_grid = np.meshgrid(range(img_size[0]), range(img_size[1]))
    y_power1_image = y_grid * image
    x_power1_image = x_grid * image
    ## raw moments
    m00 = np.sum(image)
    m10 = np.sum(y_power1_image)
    m01 = np.sum(x_power1_image)
    if m00 == 0:
        mY = (img_size[1]-1) / 2  # the crop is happening in PIL system (so inverse of numpy (x,y))
        mX = (img_size[0]-1) / 2
    else:
        mY = m10 / m00
        mX = m01 / m00
    # if crop in spherical image
    padding_size = round(max(patch_size[0]/2, patch_size[1]/2))
    spheric_pad = SphericPad(padding_size=padding_size)
    mX += padding_size
    mY += padding_size
    to_PIL = transforms.ToPILImage()
    to_Tensor = transforms.ToTensor()
    
    j = int(mX - patch_size[0]/2) 
    i = int(mY - patch_size[1]/2) 
    w = patch_size[0]
    h = patch_size[1]
    x = spheric_pad(x.view(1, x.size(0), x.size(1), x.size(2))).squeeze(0)
    x = to_PIL(x)
    patch = transforms.functional.crop(x, i, j, h, w)
    patch = to_Tensor(patch)
        
    return patch

def random_crop_preprocess(x, ratio = 2):
    '''
    arg: x, tensor 1xHxW 
    '''    
    if ratio==1:
        return x
    
    img_size = (x.size(1), x.size(2))
    patch_size = (img_size[0]/ratio, img_size[1]/ratio) 
    random_crop_transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(patch_size),transforms.ToTensor()])
    
    # set the seed as mX*mY for reproducibility (mY and mX describe the position of the centroid of the image)
    image = x[0].numpy()
    x_grid, y_grid = np.meshgrid(range(img_size[0]), range(img_size[1]))
    y_power1_image = y_grid * image
    x_power1_image = x_grid * image
    ## raw moments
    m00 = np.sum(image)
    m10 = np.sum(y_power1_image)
    m01 = np.sum(x_power1_image)
    if m00 == 0:
        mY = (img_size[1]-1) / 2
        mX = (img_size[0]-1) / 2
    else:
        mY = m10 / m00
        mX = m01 / m00
    ## raw set seed
    global_rng_state = random.getstate()
    local_seed = mX*mY
    random.seed(local_seed)

    
    n_trials = 0
    best_patch_activation = 0
    selected_patch = False
    
    activation = m00 / (img_size[0]*img_size[1]) 
    while 1:
        patch = random_crop_transform(x)
        patch_activation = patch.sum(dim=-1).sum(dim=-1) / (patch_size[0]*patch_size[1])
        
        if patch_activation > (activation * 0.5):
            selected_patch = patch
            break
        
        if patch_activation >= best_patch_activation:
            best_patch_activation = patch_activation
            selected_patch = patch
        
        n_trials +=1
        if n_trials == 20:
            break
        
    ## reput global random state
    random.setstate(global_rng_state)
        
    return selected_patch