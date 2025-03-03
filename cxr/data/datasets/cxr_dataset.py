from pathlib import Path 
import pandas as pd 
import torch.utils.data as data 
import torch  
from torchvision import transforms as T
import torch.nn as nn
import numpy as np 
from tqdm import tqdm
import pydicom 
from PIL import Image 
from concurrent.futures import ThreadPoolExecutor

class CXR_Dataset(data.Dataset):
    PATH_ROOT = Path('/mnt/ocean_storage/data/UKA/UKA_Thorax/public_export')
    LABELS = [
        'HeartSize', 
        'PulmonaryCongestion',
        'PleuralEffusion_Left', 'PleuralEffusion_Right',
        'PulmonaryOpacities_Left', 'PulmonaryOpacities_Right', 
        'Atelectasis_Left', 'Atelectasis_Right'
    ]

    def __init__(
            self,
            path_root=None,
            fold = 0,
            split= None,
            fraction=None,
            label=None, # None = all labels, list of labels or single label
            transform = None,
            random_hor_flip = False,
            random_ver_flip = False,
            random_center=False,
            random_rotate=False,
            random_inverse=False,
            cache_images=False
        ):
        self.path_root = Path(self.PATH_ROOT if path_root is None else path_root)
        self.split = split
        self.label = label if label is not None else self.LABELS
        self.cache_images = cache_images

        if transform is None: 
            self.transform = T.Compose([
                # T.Resize(448, max_size=512),
                T.RandomCrop((448, 448), pad_if_needed=True, padding_mode='constant', fill=0) if random_center else T.CenterCrop((448, 448)),
                T.Lambda(lambda x: x.transpose(1, 2) if torch.rand((1,),)[0]<0.5 else x ) if random_rotate else T.Lambda(lambda x: x),
                T.RandomHorizontalFlip() if random_hor_flip else nn.Identity(),
                T.RandomVerticalFlip() if random_ver_flip else nn.Identity(),
                T.Lambda(lambda x:-x if torch.rand((1,),)[0]<0.5 else x) if random_inverse else T.Lambda(lambda x: x),
            ])         
        else:
            self.transform = transform


        # Get split  
        df = self.load_split(self.path_root/'metadata/split.csv', fold=fold, split=split, fraction=fraction)
        
        # Merge with labels 
        df_labels = pd.read_csv(self.path_root/'metadata/annotations.csv')
        # df = df.merge(df_labels, on=['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID'], how='inner')
        df = df.merge(df_labels, on='UID', how='inner')

        self.item_pointers = df.index.tolist()
        self.df = df

        if cache_images:
            self.images = {}
            path_folder = self.path_root/"data_png_resize_512"
            with ThreadPoolExecutor(100) as executor:
                uid_to_future = {uid: executor.submit(self.load_img, path_folder / f'{uid}.png') for uid in  tqdm(self.df['UID'],  desc="Submitting tasks")}
                
                for uid, future in tqdm(uid_to_future.items(), total=len(uid_to_future)):
                    images = future.result()
                    self.images[uid] = images


    def __len__(self):
        return len(self.item_pointers)

    def load_img(self, path_img):
        # ds = pydicom.dcmread(path_img)
        # img = ds.pixel_array.astype(np.float32) # [H, W]
        img = Image.open(path_img)
        img = np.array(img).astype(np.float32)
        return img

    def __getitem__(self, index):
        idx = self.item_pointers[index]
        item = self.df.loc[idx]
        uid = item['UID']
        label = np.stack(item[self.label].values) if isinstance(self.label, list) else item[self.label] 

        label = (label > 1).astype(int)

        # static_path_data = "data"
        # rel_path_series = Path(item['PatientID'])/item['StudyInstanceUID']/item['SeriesInstanceUID']
        # filename = item['Filename']
        # path_file = self.path_root/static_path_data/rel_path_series/filename

        static_path_data = "data_png_resize_512"
        filename = f'{uid}.png'
        path_file = self.path_root/static_path_data/filename

        if self.cache_images:
            img = self.images[uid]
        else:
            img = self.load_img(path_file)
        
        img = torch.from_numpy(img)[None] # [1, H, W]

        # mask = (img>img.quantile(q=0.025)) & (img<img.quantile(q=0.975))
        # img = (img-img[mask].mean())/img[mask].std()

        img = (img-img.mean())/img.std()

        img = self.transform(img)
        
        return {'uid':uid, 'source':img, 'target':label }



    @classmethod
    def load_split(cls, filepath_or_buffer=None, fold=0, split=None, fraction=None):
        df = pd.read_csv(filepath_or_buffer)
        df = df[df['Fold'] == fold]
        if split is not None:
            df = df[df['Split'] == split]   
        if fraction is not None:
            df = df.sample(frac=fraction, random_state=0).reset_index()
        return df
    