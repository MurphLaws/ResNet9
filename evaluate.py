from fastai.data.external import untar_data, URLs
import os 


data_dir = untar_data(URLs.CIFAR)
data_dir = str(data_dir)


print(os.listdir(data_dir))
classes = os.listdir(data_dir + "/train")
print(classes)
