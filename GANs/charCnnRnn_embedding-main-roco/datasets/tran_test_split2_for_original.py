import os
from shutil import copyfile
import torch
from torchvision import transforms
import cv2
import json
from progressbar import ProgressBar

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
    #if image is None:
    #    print(img)
    #    image=cv2.imread("default.jpg",cv2.IMREAD_UNCHANGED)
    if len(image.shape) == 3 and image.shape[2] != 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    # if image is None:
    #     return
    try:
        sized_img = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)
        if len(sized_img.shape) == 3 and sized_img.shape[2] != 1:
            sized_img = cv2.cvtColor(sized_img, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(dest_path, sized_img)
    except:
        print("error in image",src_path)
        raise Exception("error")



text_root_folder=r"D:\数据集\医学数据集\roco-datasets\processed_data"
image_root_folder=r"D:\数据集\医学数据集\roco-datasets\med_images_single_cat"

target_dataset_folder="original"

import pickle
# train files
train_file_list=pickle.load(open(text_root_folder+"/train/filenames.pickle","rb"), encoding='latin1')

list_json=[]

progress=ProgressBar()

for fi in progress(range(len(train_file_list))):
    fname=train_file_list[fi]
    fname=fname+".jpg"
    image_path = image_root_folder + "/images/" + fname
    text_path = text_root_folder + "/text_c10/" + fname.replace(".jpg", ".txt")

    target_original_path=target_dataset_folder+ "/train/" + fname.split("/")[1]

    if not os.path.exists(image_path):
        continue
    if not os.path.exists(text_path):
        continue

    #if os.path.exists(target_original_path):
    #    continue

    if not os.path.exists(target_original_path):
        copyfile(image_path, target_original_path)
    if not os.path.exists(target_original_path.replace(".jpg", ".txt")):
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

progress=ProgressBar()

for fi in progress(range(len(test_file_list))):
    fname=test_file_list[fi]
    fname=fname+".jpg"
    image_path = image_root_folder + "/images/" + fname
    text_path = text_root_folder + "/text_c10/" + fname.replace(".jpg", ".txt")

    target_original_path = target_dataset_folder + "/test/" + fname.split("/")[1]



    #if os.path.exists(target_original_path):
    #    continue
    if not os.path.exists(target_original_path):
        copyfile(image_path, target_original_path)
    if not os.path.exists(target_original_path.replace(".jpg", ".txt")):
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


