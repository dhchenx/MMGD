
# load semtypes
list_class_tag=[]
list_class_names=[]

dict_image_class={}

def load_image_class(path):
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

        # save images to sub directories
        class_name_standard = class_name.replace(", ", "_").replace(" ", "_")
        dict_image_class[str(id)] = class_name_standard


        line = f_train_semtypes.readline()

    f_train_semtypes.close()

list_image_ids=[]

def load_image_file_path(path):
    '''
    f_image_files_names= open(path, encoding='utf-8')
    line = f_image_files_names.readline()
    while line:
        print(line)
        print("------------------")
        line = f_image_files_names.readline()
    '''
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
            list_image_ids.append([id,file_name])

train_semtypes_path="all_data/train/radiology/semtypes.txt"
test_semtypes_path="all_data/test/radiology/semtypes.txt"
print("loading semtypes...")
load_image_class(train_semtypes_path)
load_image_class(test_semtypes_path)

path_train="all_data/train/radiology/traindata.csv"
path_test="all_data/test/radiology/testdata.csv"
load_image_file_path(path_train)
load_image_file_path(path_test)

print("saving images.txt...")
f_img_cls_out=open("med_images_single_cat/images.txt",'w',encoding='utf-8')
for idx,img in enumerate(list_image_ids):
    if str(img[0]) not in dict_image_class.keys():
        continue
    f_img_cls_out.write(str(img[0])+' '+str(dict_image_class[str(img[0])]) +"/"+str(img[1])+"\n")
f_img_cls_out.close()

