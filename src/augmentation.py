import torch 
import torchvision.transforms as transforms 


def basic(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        #transforms.CenterCrop(args.input_size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

def simple(size,isize):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        trnasforms.Resize((isize,isize))
    ])
    gt_transform = transforms.Compose([
        transforms.ToTensor(),
        trnasforms.Resize((isize,isize))
    ])
    return data_transform,gt_transform
    