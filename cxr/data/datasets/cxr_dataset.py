from pathlib import Path 
import pandas as pd 
import torch.utils.data as data 
import torch  
from torchvision import transforms as T
import torchvision.transforms.functional as Tf 
import torch.nn as nn
import numpy as np 
from tqdm import tqdm
import pydicom 
from PIL import Image 
from concurrent.futures import ThreadPoolExecutor

class OneOf(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms
        
    def forward(self, x):
        # Randomly select one transform from the list
        t_idx = torch.randint(0, len(self.transforms), (1,)).item()
        return self.transforms[t_idx](x)

class ResizeLongEdge(nn.Module):
    def __init__(self, long_edge=448, interpolation=Image.BILINEAR, antialias=True):
        super().__init__()
        self.long_edge = long_edge
        self.interpolation = interpolation
        self.antialias = antialias

    def __call__(self, img):
        if isinstance(img, Image.Image):  # PIL Image
            w, h = img.size  
        elif isinstance(img, torch.Tensor):  # PyTorch Tensor
            h, w = img.shape[-2:]  
        else:
            raise TypeError(f"Expected PIL Image or Tensor, but got {type(img)}")

        if w > h:
            new_w, new_h = self.long_edge, int(h * (self.long_edge / w))
        else:
            new_w, new_h = int(w * (self.long_edge / h)), self.long_edge
        return Tf.resize(img, (new_h, new_w), self.interpolation, antialias=self.antialias)

    def __repr__(self):
        return f"{self.__class__.__name__}(long_edge={self.long_edge}, interpolation={self.interpolation})"


class CXR_Dataset(data.Dataset):
    PATH_ROOT = Path('/mnt/ocean_storage/data/UKA/UKA_Thorax/public_export')
    # CLASS_LABELS = {
    #     'HeartSize': ['Normal', 'Borderline', 'Enlarged', 'Massively'],
    #     'PulmonaryCongestion':  ['None', 'Questionable', 'Mild', 'Moderate', 'Severe'], 
    #     'PleuralEffusion_Left': ['None', 'Questionable', 'Mild', 'Moderate', 'Severe'],
    #     'PleuralEffusion_Right': ['None', 'Questionable', 'Mild', 'Moderate', 'Severe'],
    #     'PulmonaryOpacities_Left': ['None', 'Questionable', 'Mild', 'Moderate', 'Severe'],
    #     'PulmonaryOpacities_Right': ['None', 'Questionable', 'Mild', 'Moderate', 'Severe'],
    #     'Atelectasis_Left': ['None', 'Questionable', 'Mild', 'Moderate', 'Severe'],
    #     'Atelectasis_Right': ['None', 'Questionable', 'Mild', 'Moderate', 'Severe'],
    # }
    CLASS_LABELS = {
        'HeartSize': ['Normal', 'Borderline', 'Enlarged', 'Massively'],
        'PulmonaryCongestion':  ['None', '(+)', '+', '++', '+++'], 
        'PleuralEffusion_Left': ['None', '(+)', '+', '++', '+++'],
        'PleuralEffusion_Right': ['None', '(+)', '+', '++', '+++'],
        'PulmonaryOpacities_Left': ['None', '(+)', '+', '++', '+++'],
        'PulmonaryOpacities_Right': ['None', '(+)', '+', '++', '+++'],
        'Atelectasis_Left': ['None', '(+)', '+', '++', '+++'],
        'Atelectasis_Right': ['None', '(+)', '+', '++', '+++'],
    }


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
            cache_images=False,
            regression=False
        ):
        self.path_root = Path(self.PATH_ROOT if path_root is None else path_root)
        self.split = split
        if label is None:
            self.label = list(self.CLASS_LABELS.keys())
        elif isinstance(label, str):
            self.label = [label]
        else:
            self.label = label
        # self.class_labels_num = [len(self.CLASS_LABELS[l])-1 for l in self.label] # Remove -1 for CORN 
        self.class_labels_num = [len(self.CLASS_LABELS[l]) for l in self.label] 

        self.cache_images = cache_images
        self.regression = regression

        if transform is None: 
            self.transform = T.Compose([
                ResizeLongEdge(448),
                T.RandomCrop((448, 448), pad_if_needed=True, padding_mode='constant', fill=0) if random_center else T.CenterCrop((448, 448)),
                OneOf([
                    T.Lambda(lambda x: x.transpose(1, 2) if torch.rand((1,),)[0]<0.5 else x ) if random_rotate else T.Lambda(lambda x: x),
                    T.RandomVerticalFlip() if random_ver_flip else nn.Identity(),
                ]),
                T.RandomHorizontalFlip() if random_hor_flip else nn.Identity(),
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

    def load_item(self, item):
        uid = item['UID']
        label = np.stack(item[self.label].values) # if len(self.label)>1 else item[self.label[0]] 

        if not self.regression:
            label = (label > 1).astype(int) # WARNING: Assumes no missing or "-1" labels 

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

        #mask = (img>img.quantile(q=0.025)) & (img<img.quantile(q=0.975))
        # mask = (img>img.min()) & (img<img.max())
        # img = (img-img[mask].mean())/img[mask].std()


        img = self.transform(img)

        # img = (img-img.mean())/img.std()
        mask = (img>img.min()) & (img<img.max())
        img = (img-img[mask].mean())/img[mask].std()
        
        return {'uid':uid, 'source':img, 'target':label }


    def __getitem__(self, index):
        idx = self.item_pointers[index]
        item = self.df.loc[idx]
        return self.load_item(item)



    @classmethod
    def load_split(cls, filepath_or_buffer=None, fold=0, split=None, fraction=None):
        df = pd.read_csv(filepath_or_buffer)
        df = df[df['Fold'] == fold]
        if split is not None:
            df = df[df['Split'] == split]   
        if fraction is not None:
            df = df.sample(frac=fraction, random_state=0).reset_index()
        return df
    