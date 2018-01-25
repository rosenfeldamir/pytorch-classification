import torch
import numpy as np
from numpy import array
from torch.autograd import Variable

def makeMatchingMatrices(matching):
    # multiply the targets and units by matrices to reorder the corresponding columns to match.
    source_neurons = array(matching['source_units'])
    target_concepts = array(matching['target_concepts']).astype(int)
    my_source_neurons = source_neurons
    my_target_concepts = array(target_concepts)

    neuron_selection = np.ones(my_source_neurons.shape, np.bool)
    source_sel = np.zeros((1024, len(my_source_neurons[neuron_selection])))
    for i, n in enumerate(my_source_neurons[neuron_selection]):
        source_sel[n, i] = 1

    source_sel = Variable(torch.from_numpy(source_sel).float().cuda())
    source_sel.requires_grad = False

    target_sel = np.zeros((1197, len(my_target_concepts[neuron_selection])))
    for i, n in enumerate(my_target_concepts[neuron_selection]):
        target_sel[n, i] = 1

    target_sel = Variable(torch.from_numpy(target_sel).float().cuda())
    target_sel.requires_grad = False

    return source_sel, target_sel

def getModelOutput(model, input_var, source_sel):
    output, output0, output1, output2, output3, output4 = model(input_var)
    output0, output1, output2, output3, output4 = output0.squeeze(), output1.squeeze(), output2.squeeze(), output3.squeeze(), output4.squeeze()
    output_ = torch.cat([output0, output1, output2, output3, output4], 1)
    output_ = torch.mm(output_, source_sel)
    return output_


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


from torch.utils.data import Dataset, DataLoader


class ListDataset(Dataset):
    """An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, paths, label_matrix, img_set, split='train', transform=None):
        assert split in ['train', 'val']
        # p1=[]
        # l1=[]
        # label_matrix = label_matrix.astype(uint8)

        S = array([True if s == split else False for s in img_set]).astype(np.bool)
        self.paths = paths[S]
        self.labels = (label_matrix[S]).astype(uint8)
        self.length = len(self.paths)
        self.transform = transform

    def __getitem__(self, index):
        # load image
        # transform it
        # return it with target
        img = pil_loader(self.paths[index])
        if self.transform is not None:
            img = self.transform(img)
        target = torch.from_numpy(self.labels[index]).long()
        return img, target

    def __len__(self):
        return self.length


from torchvision import transforms
from PIL import Image


def ton(v):
    if type(v) is Variable:
        v = v.data
    return v.cpu().numpy()

def makeDataLoader(paths, img_to_concepts, img_to_split, split='train', shuffle=True, batch_size=32):
    # make the training loader.

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transform = transforms.Compose([transforms.ToTensor(), normalize])
    data = ListDataset(paths, img_to_concepts, img_to_split, split)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader

