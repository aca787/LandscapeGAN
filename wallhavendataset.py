import os 
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms, utils
from skimage import io,transform
import matplotlib.pyplot as plt
import numpy as np
import glob
# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, label = \
            sample_batched['right_images'], sample_batched['class']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    print(label)
    plt.show()

class Text2ImageDataset(Dataset):

    def __init__(self, root, embed_dim=4096, transform=None, split=0):
        self.root = root
        self.transform = transform
        self.dataset = None
        self.dataset_keys = None
        self.embed_dim = embed_dim
        
        self.split = 'train' if split == 0 else 'valid' if split == 1 else 'test'
        
        self._load_dataset(root)
        
    def _label_to_sent(self, labels):
        sentence = []
        labels = labels.split(',')
        for l in labels:
            if l  in ['<s>', '\n','</s>']:
                continue
            if ' ' not in l:
                sentence.append(l)
            else:
                sentence+=l.strip().split(' ')
        return sentence
    
    def _load_dataset(self, root):
        self.label_files = [f for f in glob.glob(root +'/**/labels.txt', recursive=True)]
        
        self.dataset = []    
        self.classes = set()
        for file_name in self.label_files:
            f = open(file_name, 'r')
            #all_data += f.read()
            for line in f:
                sample={}
                if("429 Too Many Requests" in line):
                    continue
                image_name = line.split('|')[0].strip()
                sentence = self._label_to_sent(line.split('|')[1])
                
                sample['image_path'] = os.path.join(os.path.dirname(file_name), image_name + ".jpg")
                sample['class'] = os.path.basename(os.path.dirname(file_name))
                sample['label_text'] = ' '.join(sentence)
                
                self.classes.add(sample['class'])
                self.dataset.append(sample)      
        self.classes = list(self.classes)  
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.dataset is None:
            self.dataset = self._load_dataset()

        # pdb.set_trace()
        example = self.dataset[idx]
        right_image = io.imread(example['image_path'])
        wrong_label =  np.array(self.find_wrong_label(example['label_text'])).astype(str)
        txt = np.array(example['label_text']).astype(str)
        class_ = np.array(example['class']).astype(str)

        sample = {
                'right_images':right_image,
                'wrong_label': str(wrong_label),
                'txt': str(txt),
                'class': str(class_),
                'embedding':torch.zeros(self.embed_dim),
                'wrong_embedding': torch.zeros(self.embed_dim)
                 }
        if self.transform:
            sample = self.transform(sample)

        return sample

    def find_wrong_label(self, category):
        idx = np.random.randint(len(self.classes))
        _category = self.classes[idx]

        if _category != category:
            return _category

        return self.find_wrong_label(category)


    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def _resize(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        return transform.resize(image,(new_h, new_w))

    def __call__(self, sample):
        right_image = sample['right_images']
        sample['right_images']=self._resize(right_image)
        
        return sample


class CenterCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def crop(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        return image[0: new_h,
                      0: new_w]

    def __call__(self, sample):
        right_image = sample['right_images']
        sample['right_images']=self.crop(right_image)

        return sample
        
class EmbedLabel(object):
    """ Embed the labels from dataset """
    def __init__(self, embedder):
        self.embedder=embedder
    def __call__(self, sample):
        
        sample['embedding'] = self.embedder.embed(sample['class'])
        sample['wrong_embedding'] = self.embedder.embed(sample['wrong_label'])
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        right_image = sample['right_images']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        sample['right_images'] = torch.FloatTensor(right_image.transpose((2, 0, 1)))
        sample['embedding'] = torch.FloatTensor(sample['embedding']).squeeze()
        sample['wrong_embedding'] = torch.FloatTensor(sample['wrong_embedding']).squeeze()
        return sample