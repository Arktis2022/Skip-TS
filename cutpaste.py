import random
import numpy as np
from torchvision import transforms
transform = None
def crop(image,left,top,right,bottom):
        return image[:,:,left:right,top:bottom]

def paste(image,patch, paste_left, paste_top):
        image[:,:,paste_left:paste_left+patch.size()[-2],paste_top:paste_top+patch.size()[-1]] = patch
        return image   
         
def crop_and_paste_patch(image, patch_w, patch_h, transform, rotation=False):
        """
        Crop patch from original image and paste it randomly on the same image.

        :image: [PIL] _ original image
        :patch_w: [int] _ width of the patch
        :patch_h: [int] _ height of the patch
        :transform: [binary] _ if True use Color Jitter augmentation
        :rotation: [binary[ _ if True randomly rotates image from (-45, 45) range

        :return: augmented image
        """
        #print(patch_w, patch_h)
        org_w, org_h = image.size()[-1],image.size()[-2]
        #print (org_w,org_h)
        mask = None

        patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
        patch = crop(image,patch_left, patch_top, patch_right, patch_bottom)
        #print(patch.size())
        if transform:
            patch= transform(patch)

        if rotation:
            random_rotate = random.uniform(*rotation)
            patch = patch.convert("RGBA").rotate(random_rotate, expand=True)
            mask = patch.split()[-1]

        # new location
        paste_left, paste_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        aug_image = image
        #aug_image.paste(patch, (paste_left, paste_top), mask=mask)
        aug_image = paste(aug_image,patch,paste_left,paste_top)
        return aug_image
        
def cutpaste(image, area_ratio = (0.02, 0.15), aspect_ratio = ((0.3, 1) , (1, 3.3))):
        '''
        CutPaste augmentation

        :image: [PIL] - original image
        :area_ratio: [tuple] - range for area ratio for patch
        :aspect_ratio: [tuple] -  range for aspect ratio

        :return: PIL image after CutPaste transformation
        '''
        #print(image.size()[-1],image.size()[-2])
        img_area = image.size()[-1] * image.size()[-2]
        patch_area = random.uniform(*area_ratio) * img_area
        patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
        patch_w  = int((np.sqrt(patch_area*patch_aspect))/2)
        patch_h = int(np.sqrt(patch_area/patch_aspect))
        cutpaste = crop_and_paste_patch(image, patch_w, patch_h, transform, rotation = False)
        return cutpaste
        
def cutpaste_scar(image, width = [2,16], length = [10,25], rotation = (-45, 45)):
        '''

        :image: [PIL] - original image
        :width: [list] - range for width of patch
        :length: [list] - range for length of patch
        :rotation: [tuple] - range for rotation

        :return: PIL image after CutPaste-Scare transformation
        '''
        patch_w, patch_h = random.randint(*width), random.randint(*length)
        cutpaste_scar = crop_and_paste_patch(image, patch_w, patch_h, transform, rotation = rotation)
        return cutpaste_scar
        
