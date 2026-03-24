import numpy as np
import random
import torch

class OcclusionGenerator:
    def __init__(self, mode='box', max_size=0.4):
        self.mode = mode # 'box' or 'mask'
        self.max_size = max_size

    def __call__(self, img_tensor):
        '''
        img_tensor: torch tensor of shape (C, H, W) in range [-1, 1]
        '''
        img = img_tensor.clone()
        _, h, w = img.shape
        
        if self.mode == 'box':
            box_h = int(h * random.uniform(0.1, self.max_size))
            box_w = int(w * random.uniform(0.1, self.max_size))
            
            top_y = random.randint(0, h - box_h)
            # Favor face regions (bottom half for mask, middle for eyes) when valid range exists
            if random.random() > 0.5 and h // 2 <= h - box_h:
                top_y = random.randint(h // 2, h - box_h)
                
            left_x = random.randint(0, w - box_w)
            
            img[:, top_y:top_y+box_h, left_x:left_x+box_w] = -1.0 # black out

        elif self.mode == 'mask':
            # Block out the entire lower half of the face where a medical mask usually is
            top_y = int(h * 0.45) # Start around the nose area
            
            img[:, top_y:h, :] = -1.0 # Black out everything below the nose across the entire width
        return img
