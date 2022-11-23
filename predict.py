import numpy as np
import torch
import torchvision 
import argparse
from torch import nn
import pickle
from PIL import Image
sized_image =[225 ,225]
def load_model(model_arc , model_pth):
    model = pickle.load(open(model_arc , mode="rb"))
    checkpoint = torch.load(model_pth)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

test_transform =  torchvision.transforms.Compose([
                                                 
                                                  torchvision.transforms.RandomResizedCrop(224),
                                                  torchvision.transforms.Resize(sized_image),
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std =[0.229, 0.224, 0.225]),
                                                    ])
def process_image(image):
    
    image = Image.open(image)
    image = test_transform(image)
   
    return image
device ="cuda" if torch.cuda.is_available() else "cpu"
def predict(image_path, model, checkpoint , cat_to_name,topk=5,):
    image = process_image(image_path)
    model.eval()
    prediction = model(image.unsqueeze(0).float().to(device))
    value , indices = prediction.topk(topk)
    data_indices =  checkpoint['class_to_idx']
    labels =[]
    for indx in indices.cpu().numpy()[0]:
        for i , j in data_indices.items():
            if j==indx:
                labels.append(cat_to_name.get(i))
    return labels , value
def main():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--image-path", type=str , help="image path")
    arg_parse.add_argument("--check-point", type=str, help="check point file ")
    arg_parse.add_argument("--top-k", type=int , default=5 , help="top of most predictors")
    arg_parse.add_argument("--category-names", type=str)
    arg_parse.add_argument("--gpu", type=bool , default=True)
    args = arg_parse.parse_args()
    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    import os
    model = load_model(os.path.join("saved_models", "model_ark.pkl"), os.path.join("saved_models",args.check_point+".pth"))
    checkpoint = torch.load(os.path.join("saved_models",args.check_point+".pth"))
    labels , values = predict(args.image_path , model ,checkpoint ,args.category_names , args.top_k)
    print(f"Labels {labels} values {values}")
if __name__ == "__main__":
    main()