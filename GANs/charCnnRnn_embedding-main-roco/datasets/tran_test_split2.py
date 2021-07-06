import os
from shutil import copyfile
import torch
from torchvision import transforms
import cv2
import json

from models.conv_autoencoder.models import ConvAutoencoder
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

def scale_img_save(src_path, dest_path,dsize):
    # Create 256x256 and 64x64 gray copies

    image = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    # resize image to 64X64 and save image
    # dsize = (64, 64)
    if image is None:
        return

    sized_img = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)
    if len(sized_img.shape) == 3 and sized_img.shape[2] != 1:
        sized_img = cv2.cvtColor(sized_img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(dest_path, sized_img)


text_root_folder=r"D:\数据集\医学数据集\roco-datasets\processed_data"
image_root_folder=r"D:\数据集\医学数据集\roco-datasets\med_images_single_cat"
import pickle
# train files
train_file_list=pickle.load(open(text_root_folder+"/train/filenames.pickle","rb"), encoding='latin1')

list_json=[]

for fname in train_file_list:
    fname=fname+".jpg"
    image_path = image_root_folder + "/images/" + fname
    text_path = text_root_folder + "/text_c10/" + fname.replace(".jpg", ".txt")

    target_original_path="train/" + fname.split("/")[1]

    if os.path.exists(target_original_path):
        continue

    copyfile(image_path, target_original_path)
    copyfile(text_path, target_original_path.replace(".jpg", ".txt"))
    # scale to 64
    scale_img_save( target_original_path,target_original_path,dsize=(64,64))
    # encode image
    enc_img=encode_img(target_original_path)
    torch.save(enc_img, 'enc_64x64_images/'+fname.split("/")[1].replace(".jpg","")+'.pt')
    # obtain text
    enc_text=open(text_path,'r',encoding='utf-8').readlines()
    dict_model = {}
    dict_model['text'] =(' '.join(enc_text)).replace("\n","")
    dict_model["encod_64x64_path"] = '/enc_64x64_images/'+ fname.split("/")[1].replace(".jpg","")+'.pt'
    list_json.append(dict_model)

# test files
test_file_list=pickle.load(open(text_root_folder+"/test/filenames.pickle","rb"), encoding='latin1')

for fname in test_file_list:
    fname=fname+".jpg"
    image_path = image_root_folder + "/images/" + fname
    text_path = text_root_folder + "/text_c10/" + fname.replace(".jpg", ".txt")

    target_original_path = "test/" + fname.split("/")[1]

    if os.path.exists(target_original_path):
        continue

    copyfile(image_path, target_original_path)
    copyfile(text_path, target_original_path.replace(".jpg", ".txt"))
    # scale to 64
    scale_img_save(target_original_path, target_original_path,dsize=(64,64))
    # encode image
    enc_img = encode_img(target_original_path)
    torch.save(enc_img, 'enc_64x64_images/' + fname.split("/")[1].replace(".jpg", "") + '.pt')
    # obtain text
    enc_text = open(text_path,'r',encoding='utf-8').readlines()
    dict_model = {}
    dict_model['text'] = (' '.join(enc_text)).replace("\n","")
    dict_model["encod_64x64_path"] = '/enc_64x64_images/' +  fname.split("/")[1].replace(".jpg", "") + '.pt'
    list_json.append(dict_model)

json.dump(list_json,open("images_data.json",'w',encoding='utf-8'))


