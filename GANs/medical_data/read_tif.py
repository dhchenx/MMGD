from skimage import io as io
img = io.imread(r"D:\MMML\ct-medical-archive\tiff_images\ID_0019_AGE_0070_CONTRAST_1_CT.tif")
io.imshow(img)
io.show()