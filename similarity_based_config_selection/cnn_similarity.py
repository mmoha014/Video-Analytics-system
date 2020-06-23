import torch
import torchvision
from torchvision import transforms
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets
import time
import os
import matplotlib.pyplot as plt
from PIL import Image
import re
from imutils import paths
import pickle


class VGG19_features(nn.Module):
    def __init__(self, trained_model):
        super(VGG19_features,self).__init__()
        self.features = nn.Sequential(*list(trained_model.children())[:-1])

    def forward(self,x):
        x = self.features(x)
        return x
def main(dotrain):
    plt.ion()
    num_epochs=10

    data_transforms = {
        'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    ############################ Load Data #########
    # data_dir = 'hymenoptera_data'
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
    #                                          data_transforms[x])
    #                   for x in ['train', 'val']}
    # dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=10,
    #                                                  shuffle=True, num_workers=1)
    #                   for x in ['train', 'val']}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # class_names = image_datasets['train'].classes
    # ############################ Load Model ##################
    # model_conv = torchvision.models.vgg19(pretrained=True)
    # use_gpu = torch.cuda.is_available()

    # train_total_data = len(image_datasets['train'])
    # val_total_data = len(image_datasets['val'])

    # if dotrain:
    #     for param in model_conv.parameters():
    #         param.requires_grad = False
    #     num_ftrs = model_conv.fc.in_features
    #     model_conv.fc = torch.nn.Linear(num_ftrs,2)
    
    #     if use_gpu:
    #         model_conv = model_conv.cuda()
    #     #I ignored the rest of code
    # else:
    if True:
        trained_model = torchvision.models.vgg19(pretrained=True)
        trained_model.eval()
        model_urls = {
            'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
            'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
            'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
            'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
            'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
            'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
            'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
            'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
            }
        # trained_model = torch.load("vgg19-dcbb9e9d.pth")#"'vgg19_bn-c79401a0.pth')
        mymodel = VGG19_features(trained_model)
        loader = transforms.Compose([transforms.ToTensor()])
        numbers = re.compile(r'(\d+)')

        def numericalSort(value):
            parts = numbers.split(value)
            parts[1::2] = map(int, parts[1::2])
            return parts
        
        vdo = '2'
        capt = sorted(paths.list_images('processed_frames/'+vdo), key = numericalSort)
        segm = 0
        extracted_features = list()
        while segm<len(capt):
            img = Image.open(capt[segm])#cv2.imread('/home/morteza/Videos/traffic camera/keyframes/1/seg0.jpg')            
            img = img.resize((960,960))
            img = loader(img).float()
            img = Variable(img)
            img = img.unsqueeze(0)            
            # pred = trained_model(img)
            # print(pred)        
            ### Loading the model and feeding the image to get features in the layer whose output is our interest        
            output = mymodel(img)
            features = (output.data).cpu().numpy() #converting the output of the layer to the numpy array
            # features = features.flatten()        
            extracted_features.append(features.flatten())
            segm += 1
        pickle.dump( extracted_features, open( '/home/morteza/Videos/traffic camera/keyframes/newFeatures/'+vdo+'.p', "wb" ) )
if __name__ == "__main__":
    main(False)