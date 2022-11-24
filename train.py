import numpy as np
import torch
import torchvision 
import argparse
from torch import nn
sized_image =[225 ,225]
def get_train_loader(train_path):
    data_transforms = torchvision.transforms.Compose([
                                                  torchvision.transforms.RandomRotation(30),
                                                  torchvision.transforms.RandomResizedCrop(224),
                                                  torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                                  torchvision.transforms.Resize(sized_image),
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std =[0.229, 0.224, 0.225]),
                                                    ])
    data_set = torchvision.datasets.ImageFolder(train_path , transform=data_transforms)
    trainloader = torch.utils.data.DataLoader(data_set , batch_size=32 , shuffle=True)
    return trainloader , data_set
def get_valid_loader(valid_path):
    valid_transform = torchvision.transforms.Compose([
                                                 
                                                  torchvision.transforms.RandomResizedCrop(224),
                                                  torchvision.transforms.Resize(sized_image),
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std =[0.229, 0.224, 0.225]),
                                                    ])
    data_set = torchvision.datasets.ImageFolder(valid_path , transform=valid_transform)
    validloader = torch.utils.data.DataLoader(data_set , batch_size=32)
    return validloader

def get_model(name , device , num_classes,hidden_units):
    model = torchvision.models.__dict__.get(name)
    if model:
        model = model(pretrained =True)
        for param in model.paramters():
            param.require_grad=False
    else :
        raise Exception("this model not found ")
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.LazyLinear(512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512 ,hidden_units),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(hidden_units , num_classes)
    )
    return model.to(device)
device=None
@torch.no_grad()
def accuracy(model , x_batch , y_batch):
    model.eval()
    y_pred = model(x_batch.to(device))
    _ , y_pred = y_pred.max(-1)
    correct = y_pred == y_batch
    return correct.cpu().numpy().mean()
def calc_loss(model , loss_fn , x_batch , y_batch):
    model.eval()
    y_pred = model(x_batch.to(device))
    loss = loss_fn(y_pred , y_batch)
    return loss.item()
def train(model , train_loader , test_loader , loss_fn , optimizer , epochs):
    train_loesse , validation_losses =[] , []
    train_accuracies , validation_accuracuies = [] , []
    for e in range(epochs):
        
        train_loss_epoch , validation_loss_epoch =[] , []
        train_acc_epoch , validation_acc_epoch =[] , []
        for x_batch , y_batch in train_loader:
            model.train()
            y_pred = model(x_batch.float().to(device))
            loss = loss_fn(y_pred.to(device) , y_batch.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss_epoch.append(loss.item())
            train_acc = accuracy(model , x_batch.float().to(device) , y_batch.to(device))
            train_acc_epoch.append(train_acc)
            
        for x_batch , y_batch in test_loader:
            valid_loss = calc_loss(model , loss_fn , x_batch.float().to(device) , y_batch.to(device))
            valid_acc =  accuracy(model , x_batch.to(device) , y_batch.to(device))
            validation_loss_epoch.append(valid_loss)
            validation_acc_epoch.append(valid_acc)
        train_loesse.append(np.array(train_loss_epoch).mean())
        train_accuracies.append(np.array(train_acc_epoch).mean())
        validation_losses.append(np.array(validation_loss_epoch).mean())
        validation_accuracuies.append(np.array(validation_acc_epoch).mean())
        print(f"Epoch {e+1} train loss : {train_loesse[-1]} train Acc : {train_accuracies[-1]} valid loss {validation_losses[-1]} valid Acc {validation_accuracuies[-1]}")
    return train_loesse , train_accuracies , validation_losses , validation_accuracuies

def main():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--train-data", type=str , help="train directory")
    arg_parse.add_argument("--valid-data", type=str , help="validation path")
    arg_parse.add_argument("--learning-rate", typr=float , default=0.01,metavar="N",help="learning rate")
    arg_parse.add_argument("--hidden-units", type=int , metavar="N", default=128 , help="Hidden units in nn")
    arg_parse.add_argument("--epochs", type=int , default=5, metavar="N", help="number of epochs")
    arg_parse.add_argument("--arch", type=str , default="vgg16", help="Architrcture name")
    arg_parse.add_argument("--save-dir", default="saved_models", type=str, help="path of model file")
    arg_parse.add_argument("--gpu", type=bool ,default=True , help="uisng gpu or not")
    args = arg_parse.parse_args()
    dataloader , data_set = get_train_loader(args.train_data)
    validaloader = get_valid_loader(args.valid_data)
    device = "cuda" if args.gpu else "cpu"
    num_classes = len(data_set.class_to_idx)
    model = get_model(args.arch , device , num_classes , args.hidden_units )
    optimizer = torch.optim.Adam(model.paramters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loesse , train_accuracies , validation_losses , validation_accuracuies = train(model , dataloader ,validaloader,
    loss_fn, optimizer, args.epochs)
    checkpoint = {
    'epochs': args.epochs,
    'learning_rate': args.learning_rate,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'criterion_state_dict': loss_fn.state_dict(),
    'class_to_idx' : data_set.class_to_idx
    }
    import os
    torch.save(checkpoint,os.path.join(args.save_dir , "checkpoint.pth") )
    import pickle
    pickle.dump(model , open(os.path.join(args.save_dir ,"model_arc.pkl"), mode="wb"))
    
    

if __name__ =="__main__":
    main()
