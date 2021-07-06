import os
from shutil import copyfile
# load images id
dict_image_ids={}

def load_image_file_path(path):

    import csv
    with open(path,encoding='utf-8')as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            # print(row)
            roco_id=row[0]
            if roco_id=='id':
                continue
            id=int(roco_id.split("_")[1])
            file_name=row[1]
            text=row[2].replace("\n","")
            # print('id:',id)
            # print("---------------")
            dict_image_ids[id]=file_name

dict_captions={}
def load_texts_of_image(path):
    f_caption=open(path,encoding='utf-8')
    for line in f_caption.readlines():
        line=line.strip()
        if line=="":
            continue
        fs=line.split('		')
        if len(fs)<2:
            continue
        roco_id=fs[0].strip()
        caption=fs[1].strip()
        caption=caption.replace("\t"," ")
        print(caption)
        dict_captions[roco_id]=caption
    f_caption.close()

print("loading images ids...")
path_train="all_data/train/radiology/traindata.csv"
path_test="all_data/test/radiology/testdata.csv"
load_image_file_path(path_train)
load_image_file_path(path_test)

print("loading caption ids...")
path_cap_train="all_data/train/radiology/keywords.txt"
path_cap_test="all_data/test/radiology/keywords.txt"
load_texts_of_image(path_cap_train)
load_texts_of_image(path_cap_test)

# load semtypes
list_class_tag=[]
list_class_names=[]

list_image_class_label=[]

def load_semtypes(path):
    f_train_semtypes = open(path, encoding='utf-8')
    line = f_train_semtypes.readline()
    while line:
        line = line.replace('\r', '').strip()
        if line == '':
            line = f_train_semtypes.readline()
            continue
        fs = line.split('\t')
        if len(fs) < 2:
            line = f_train_semtypes.readline()
            continue
        roco_id = fs[0]
        id=int(roco_id.split("_")[1])
        ls = fs[2:]
        if len(ls) < 2:
            line = f_train_semtypes.readline()
            continue
        # print(roco_id, ls)

        class_tag = ls[0]
        class_name = ls[0 + 1]
        if class_tag not in list_class_tag:
            list_class_tag.append(class_tag)
            list_class_names.append(class_name)
        class_id = list_class_tag.index(class_tag) + 1
        list_image_class_label.append([id, class_id])
        # save images to sub directories
        class_name_standard = class_name.replace(", ", "_").replace(" ", "_")
        sub_folder = "med_images_single_cat/images/" + class_name_standard
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder)
        source_image_path = os.path.dirname(path) + "/images/" + dict_image_ids[id]
        target_image_path = sub_folder + "/" + dict_image_ids[id]
        if os.path.exists(source_image_path):
            copyfile(source_image_path, target_image_path)

        # save text caption of image
        sub_text_folder= "processed_data/text_c10/" + class_name_standard
        if not os.path.exists(sub_text_folder):
            os.mkdir(sub_text_folder)
        target_caption=dict_captions[roco_id]
        target_text_path= sub_text_folder + "/" + dict_image_ids[id].replace(".jpg",".txt")
        f_save=open(target_text_path,'w',encoding='utf-8')
        f_save.write(target_caption)
        f_save.close()

        # read line
        line = f_train_semtypes.readline()

    f_train_semtypes.close()

train_semtypes_path="all_data/train/radiology/semtypes.txt"
test_semtypes_path="all_data/test/radiology/semtypes.txt"
print("loading semtypes...")
load_semtypes(train_semtypes_path)
load_semtypes(test_semtypes_path)

print("saving classes.txt...")
f_class_out=open("med_images_single_cat/classes.txt...",'w',encoding='utf-8')
for idx,class_name in enumerate(list_class_names):
    f_class_out.write(str(idx+1)+' '+class_name.replace(", ","_").replace(" ","_")+"\n")
f_class_out.close()

print("saving image_class_labels.txt...")
f_img_cls_out=open("med_images_single_cat/image_class_labeles.txt",'w',encoding='utf-8')
for idx,img_class in enumerate(list_image_class_label):
    f_img_cls_out.write(str(img_class[0])+' '+str(img_class[1])+"\n")
f_img_cls_out.close()

