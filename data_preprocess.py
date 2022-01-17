import os
from PIL import Image
import numpy as np
from torchvision import transforms as T
import shutil


def make_dict(file_name):
    index = {}

    with open(os.getcwd() + "/" + file_name + ".txt", 'r') as f:
        lines = f.readlines()
        new_lines = [line[:-1] for line in lines]

        for line in new_lines:
            split_line = line.split(',')
            index[split_line[0]] = split_line[1]

    return index


def make_folders(name):
    for i in range(1, 101):
        folder_name = os.getcwd()+"/"+name+"/"+str(i)
        os.mkdir(folder_name)


rotate = T.Compose([
    T.RandomRotation(degrees=(0, 45))
])

greyscale = T.Compose([
    T.Grayscale(num_output_channels=3)
])

jitter = T.Compose([
    T.ColorJitter(brightness=0.75)
])


def move_data(to_name, from_name, test_index, train_index, num):
    image_path = os.getcwd() + "/"+to_name+"/"

    for file in os.listdir(os.getcwd()+"/"+from_name+"/"+from_name):
        img = Image.open(os.path.join(os.getcwd()+"/"+from_name+"/"+from_name, file)).convert("RGB")
        if to_name == "test_data":
            img_name = file.split('.', 1)[0]
            if len(img_name) > 4:
                img_name = img_name[:-1]
            img.save(image_path+test_index[img_name]+"/"+file)
        else:
            img_name = file.split('.', 1)[0]
            if len(img_name) > 4:
                img_name = img_name[:-1]

            if num[train_index[img_name]] % 3 == 2:
                num[train_index[img_name]] = num[train_index[img_name]] + 1
                image_path = os.getcwd() + "/val_data/"

                img.save(image_path + train_index[img_name] + "/" + file)

                flip_lr = Image.fromarray(np.uint8(np.fliplr(img))).convert('RGB')
                flip_lr.save(image_path + train_index[img_name] + "/" + file.replace('.', "_flip_lr."))

                flip_ud = Image.fromarray(np.uint8(np.flipud(img))).convert('RGB')
                flip_ud.save(image_path + train_index[img_name] + "/" + file.replace('.', "_flip_ud."))

                rotate_img = rotate(img)
                rotate_img.save(image_path + train_index[img_name] + "/" + file.replace('.', "_rotate."))

                greyscale_img = greyscale(img)
                greyscale_img.save(image_path + train_index[img_name] + "/" + file.replace('.', "_grey."))

                jitter_img = jitter(img)
                jitter_img.save(image_path + train_index[img_name] + "/" + file.replace('.', "_jitter."))

            else:
                num[train_index[img_name]] = num[train_index[img_name]] + 1

                image_path = os.getcwd() + "/" + to_name + "/"

                img.save(image_path+train_index[img_name]+"/"+file)

                flip_lr = Image.fromarray(np.uint8(np.fliplr(img))).convert('RGB')
                flip_lr.save(image_path+train_index[img_name]+"/"+file.replace('.', "_flip_lr."))

                flip_ud = Image.fromarray(np.uint8(np.flipud(img))).convert('RGB')
                flip_ud.save(image_path+train_index[img_name]+"/"+file.replace('.', "_flip_ud."))

                rotate_img = rotate(img)
                rotate_img.save(image_path+train_index[img_name]+"/"+file.replace('.', "_rotate."))

                greyscale_img = greyscale(img)
                greyscale_img.save(image_path+train_index[img_name]+"/"+file.replace('.', "_grey."))

                jitter_img = jitter(img)
                jitter_img.save(image_path+train_index[img_name]+"/"+file.replace('.', "_jitter."))


if __name__=="__main__":
    train_index = make_dict("index_train")
    test_index = make_dict("index_test")

    shutil.rmtree(os.getcwd() + "/test_data")
    shutil.rmtree(os.getcwd() + "/train_data")
    shutil.rmtree(os.getcwd() + "/val_data")

    os.mkdir(os.getcwd() + "/test_data")
    os.mkdir(os.getcwd() + "/train_data")
    os.mkdir(os.getcwd() + "/val_data")

    make_folders("train_data")
    make_folders("test_data")
    make_folders("val_data")

    num = {}
    for i in range(1, 101):
        num[str(i)] = 0

    move_data("test_data", "cropTest", test_index, train_index, num)
    move_data("train_data", "cropTrain", test_index, train_index, num)
