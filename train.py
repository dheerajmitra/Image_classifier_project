# Imports here
import numpy as np
import torch
import argparse
from PIL import Image
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict

# arguments to be passed
parser = argparse.ArgumentParser(description='Flower Classification')
parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='saved checkpoint')
parser.add_argument('--gpu', type=bool, default=True, help='to use GPU')
parser.add_argument('--arch', type=str, default='vgg16', help='architecture')
parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='initial hidden units')
parser.add_argument('--epochs', type=int, default=1, help='training epochs')

args = parser.parse_args()   

data_dir = args.data_dir


train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets

train_tranforms=transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
valid_transforms= transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),    
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
test_transforms=transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),    
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])



# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir,transform=train_tranforms)
valid_data = datasets.ImageFolder(valid_dir,transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir,transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloaders = torch.utils.data.DataLoader(train_data,batch_size=32,shuffle=True)
validloaders = torch.utils.data.DataLoader(valid_data,batch_size=32,shuffle=True)
testloaders = torch.utils.data.DataLoader(test_data,batch_size=32,shuffle=True)

# TODO: Build and train your network
device =torch.device('cuda:0' if args.gpu else 'cpu')

if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)     
        num_features = model.classifier[0].in_features

elif args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        num_features = model.classifier.in_features

elif args.arch == 'resnet':
        model = models.resnet18(pretrained = True)
        num_features = model.fc.in_features
else:
        print("Im sorry but {} is not a valid model.Did you mean vgg16,densenet121,or alexnet?".format( args.arch))
        
for param in model.parameters():
    param.requires_grad=False
classifier=nn.Sequential(OrderedDict([
    ('fc1',nn.Linear(num_features,args.hidden_units)),
    ('relu',nn.ReLU()),
    ('drop',nn.Dropout(0.2)),
    ('hidden', nn.Linear(args.hidden_units, 90)),
    ('relu_1',nn.ReLU()),
     ('fc2', nn.Linear(90,102)),
    ('output',nn.LogSoftmax(dim=1))
                      
    
]))    
if args.arch=='resnet':
    model.fc = classifier
else:    
    model.classifier = classifier
criterion=nn.NLLLoss()
if args.arch=='resnet':
    optimizer=optim.Adam(model.fc.parameters(),lr=args.learning_rate)
else:
    optimizer=optim.Adam(model.classifier.parameters(),lr=args.learning_rate)

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 32
model.to(device)
for epoch in range(epochs):
    for inputs, labels in trainloaders:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloaders:
                    model.to(device)
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print("Epoch: {}/{}.. ".format(epoch+1,epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid  Loss: {:.3f}.. ".format(valid_loss/len(validloaders)),
                  "Valid Accuracy: {:.3f}".format(accuracy/len(validloaders)))       

            running_loss = 0
           # model.train()




# TODO: Do validation on the test set
test_loss = 0
accuracy = 0
model.eval()
model.to(device)
with torch.no_grad():
    for inputs, labels in testloaders:
                    
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
print("Test Loss: {:.3f}.. ".format(valid_loss/len(testloaders)),
       "Test Accuracy: {:.3f}".format(accuracy/len(testloaders)))



# TODO: Write a function that loads a checkpoint and rebuilds the model
# TODO: Save the checkpoint 
      
model.class_to_idx = train_data.class_to_idx

check_point = {'arch':args.arch,
              'input':num_features,
              'output':102,
              'epochs':args.epochs,
              'learning_rate':args.learning_rate,
              'dropout':0.2,
              'batch_size':32,
              'classifier':classifier,
              'state_dict':model.state_dict(),
              'optimizer':optimizer.state_dict(),
              'class_to_idx': model.class_to_idx}
torch.save(check_point, args.save_dir)
