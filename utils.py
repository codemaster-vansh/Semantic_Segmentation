import os
import json
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

if not os.path.exists(os.getcwd(),'color_mapping.json'):
    raise ImportError("Please store color mappings in json file in the CWD")
else:
    with open(os.path.join(os.getcwd(),'color_mapping.json'),'r') as file:
        mapping = json.load(file)
        mapping = dict(mapping)

    color_mapping = {}
    for keys, values in mapping.items():
        color_mapping[tuple(values)] = int(keys)
    
class PairedImageDataset(Dataset):
    def __init__(self,input_dir,output_dir,height,width,file_extension=".png"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.filenames = sorted(os.listdir(input_dir))
        self.file_extension = file_extension
        self.color_to_class = color_mapping

        self.input_transform = transforms.Compose([
            transforms.Resize((height,width)),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((height,width),interpolation=transforms.InterpolationMode.NEAREST)
        ])

    def __len__(self):
        return len(self.filenames)
    
    def rgb_to_class_index(self,mask_rgb):
        mask_array = np.array(mask_rgb)
        h,w = mask_array.shape[:2]
        class_mask = np.zeros((h,w),dtype=np.int_)

        for color, class_idx in self.color_to_class.items():
            color_mask = np.all(np.abs(mask_array - color) < 10,axis=2)
            class_mask[color_mask] = class_idx

        return class_mask

    def __getitem__(self, index):
        input_name = self.filenames[index]
        base_name = os.path.splitext(input_name)[0]
        output_name = base_name + self.file_extension

        input_path = os.path.join(self.input_dir,input_name)
        output_path = os.path.join(self.output_dir,output_name)

        input_image = Image.open(input_path).convert("RGB")
        output_image = Image.open(output_path).convert("RGB")

        if self.input_transform:
            input_image = self.input_transform(input_image)
        
        if self.mask_transform:
            output_image = self.mask_transform(output_image)

        class_mask = self.rgb_to_class_index(output_image)
        output_image = torch.from_numpy(class_mask).long()

        return input_image,output_image
    

class EncoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(EncoderBlock,self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        features = self.conv_block(x)
        pooled = self.pool(features)
        return pooled, features
        

class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,skip_input_channels):
        super(DecoderBlock,self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            padding=0
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels+skip_input_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self,inputs,skip_features):
        x = self.upconv(inputs)
        skip_features_resized = nn.functional.interpolate(
            skip_features,size=x.shape[2:],mode='bilinear',align_corners=False
        )
        x = torch.cat([x,skip_features_resized],dim=1)
        return self.conv_block(x)
    

class UNet(nn.Module):
    def __init__(self,num_channels):
        super(UNet,self).__init__()
        self.encoder1 = EncoderBlock(3,64)
        self.encoder2 = EncoderBlock(64,128)
        self.encoder3 = EncoderBlock(128,256)
        self.encoder4 = EncoderBlock(256,512)
        self.encoder5 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = DecoderBlock(1024,512,512)
        self.decoder2 = DecoderBlock(512,256,256)
        self.decoder3 = DecoderBlock(256,128,128)
        self.decoder4 = DecoderBlock(128,64,64)
        self.shorten = nn.Conv2d(in_channels=64,out_channels=num_channels,kernel_size=1,padding=0) # num_channels to be same as NUM_MASKS from determine_mask_mapping.py
        
    def forward(self,inputs):
        skip_features = []
        x,features1 = self.encoder1(inputs)
        skip_features.append(features1)
        x,features2 = self.encoder2(x)
        skip_features.append(features2)
        x,features3 = self.encoder3(x)
        skip_features.append(features3)
        x,features4 = self.encoder4(x)
        skip_features.append(features4)
        x = self.encoder5(x)
        x = self.decoder1(x,skip_features[-1])
        x = self.decoder2(x,skip_features[-2])
        x = self.decoder3(x,skip_features[-3])
        x = self.decoder4(x,skip_features[-4])
        return self.shorten(x)
