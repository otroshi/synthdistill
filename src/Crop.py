import torch
import numpy as np
import imp
import os

def Crop_and_resize(img, ):

    fixed_positions={'reye': ( int(480/1024*img.size(2)), int(380/1024*img.size(3))), 
                     'leye': ( int(480/1024*img.size(2)), int(650/1024*img.size(3)))}

    CROPPED_IMAGE_SIZE=(112, 112)

    cropped_positions = {
                        "leye": (51.6, 73.5318),
                        "reye": (51.6, 38.2946)
                         }

    """
    Steps:
        1) find rescale ratio

        2) find corresponding pixel in 1024 image which will be mapped to                          
        the coordinate (0,0) at the croped_and_resized image
        
        3) find corresponding pixel in 1024 image which will be mapped to                          
        the coordinate (112,112) at the croped_and_resized image
        
        4) crop image in 1024
        
        5) resize the cropped image
    """
    # step1: find rescale ratio
    alpha = ( cropped_positions['leye'][1] - cropped_positions['reye'][1] )  /  ( fixed_positions['leye'][1]- fixed_positions['reye'][1] ) 
    
    # step2: find corresponding pixel in 1024 image for (0,0) at the croped_and_resized image
    coord_0_0_at_1024 = np.array(fixed_positions['reye']) - 1/alpha* np.array(cropped_positions['reye'])
    
    # step3: find corresponding pixel in 1024 image for (112,112) at the croped_and_resized image
    coord_112_112_at_1024 = coord_0_0_at_1024 + np.array(CROPPED_IMAGE_SIZE) / alpha
    
    # step4: crop image in 1024
    cropped_img_1024 = img[:,
                           :,
                           int(coord_0_0_at_1024[0]) : int(coord_112_112_at_1024[0]),
                           int(coord_0_0_at_1024[1]) : int(coord_112_112_at_1024[1])
                           ]
    
    # step5: resize the cropped image
    # resized_and_croped_image = cv2.resize(cropped_img_1024, CROPPED_IMAGE_SIZE) 
    resized_and_croped_image = torch.nn.functional.interpolate(cropped_img_1024, mode='bilinear', size=CROPPED_IMAGE_SIZE, align_corners=False)

    
    return resized_and_croped_image
