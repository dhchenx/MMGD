import PIL
from PIL import Image,ImageOps
import os
import numpy as np

#image = Image.open('PMC29044_cc-4-4-245-1.jpg')
#img=image.resize((64,64))
#img.save('PMC29044_cc-4-4-245-1.jpg')

g = os.walk(r"original/test")
for path,dir_list,file_list in g:
    for file_name in file_list:
        if not file_name.endswith(".jpg"):
            continue
        full_path=os.path.join(path, file_name)
        # print(full_path)
        image = Image.open(full_path)
        #if image.size==(64,64):
        #    image = np.stack((image,) * 3, axis=-1)
        img=image.resize((64,64))
        if image is None:
            # print(full_path)
            print(full_path)
            raise Exception("img is None!")
        img=ImageOps.grayscale(image)
        img.save(full_path)
