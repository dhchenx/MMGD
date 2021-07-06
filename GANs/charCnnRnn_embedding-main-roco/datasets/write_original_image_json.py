import json
import os
import imageio
import torch
from  torchvision import transforms
import random
from PIL import Image
import cv2
from models.conv_autoencoder.models import ConvAutoencoder

def make_random_sentence():
  nouns = ["puppy", "car", "rabbit", "girl", "monkey"]
  verbs = ["runs", "hits", "jumps", "drives", "barfs"]
  adv = ["crazily.", "dutifully.", "foolishly.", "merrily.", "occasionally."]
  adj = ["adorable", "clueless", "dirty", "odd", "stupid"]

  random_entry = lambda x: x[random.randrange(len(x))]
  return " ".join([random_entry(nouns), random_entry(verbs), random_entry(adv), random_entry(adj)])

from sentence_generator import Generator
loader = transforms.Compose([
    transforms.ToTensor()])
def image_loader(image_name):
    image = Image.open(image_name).convert('LA')
    image = loader(image).unsqueeze(0)
    return image.to("cpu", torch.float)

def encode_img(img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = ConvAutoencoder()
    encoder.load_state_dict(torch.load("../output/conv_autoencoder.pt"))
    encoder = encoder.to(device)
    torch.no_grad()
    encoder.eval()
    transform = transforms.ToTensor()
    image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    image = transform(image)
    image = image.float()
    image = image.to(device)
    enc_img = encoder(image.unsqueeze(0), encoder_mode=True)
    return enc_img

list=[]
list_file = os.listdir("train")  # 列出文件夹下所有的目录与文件
count=1
for i in range(0, len(list_file)):
    path = os.path.join("train", list_file[i])
    # image=imageio.imread(path)
    # image=image_loader(path)

    image=encode_img(path)
    print("tensor size:",image.size())
    torch.save(image,'enc_64x64_images/'+str(count)+".pt")

    sentence=make_random_sentence()

    f_out=open("train/"+list_file[i].replace(".jpeg",".txt"),'w')
    f_out.write(sentence)
    f_out.close()


    print(path)
    dict_model={}
    dict_model['text']=sentence
    dict_model["encod_64x64_path"]='/enc_64x64_images/'+str(count)+".pt"
    list.append(dict_model)
    count = count + 1

json.dump(list,open("images_data.json",'w',encoding='utf-8'))



