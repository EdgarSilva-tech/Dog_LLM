import os
import random
import shutil
from langchain_community.retrievers import WikipediaRetriever

if not os.path.isdir("data"):
    data_dir = "data"

    # Create the directories
    os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "validation"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "full_dataset"), exist_ok=True)

    # Downloading the Dataset through the Kaggle API
    os.system("kaggle datasets download -d jessicali9530/stanford-dogs-dataset")

    # Unzipping
    os.system("tar -xf stanford-dogs-dataset.zip")

   # os.system("move annotations data/")

   # os.system("move images data/")

    os.system("del stanford-dogs-dataset.zip")

annot = "data/Annotation"
img = "images/Images"
dir_list = os.listdir(img)
data_train_dir = "data/train"
data_val_dir = "data/validation"
   
print("Files and directories in '", img, "' :")  
   
# print the list 
[os.rename(f'{img}/{dir}', f'{img}/{dir.split("-")[1]}') for dir in dir_list]
dir_list = os.listdir(img)
# for dirpath, dirnames, filenames in os.walk(path):
#     print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
#print(random.choice(dir_list))
shutil.copytree(img, "data/full_dataset")

train_list = []
val_list = []
train_pct = 0.8


for dir in dir_list:
    breed_train_dir = os.path.join(data_train_dir, dir)
    val_breed_dir = os.path.join(data_val_dir, dir)
    image_dir = os.path.join(img, dir)
    if not os.path.isdir(breed_train_dir):
        os.makedirs(breed_train_dir, exist_ok=True)
    if not os.path.isdir(val_breed_dir):
        os.makedirs(val_breed_dir, exist_ok=True)

    num_images = int(len(os.listdir(image_dir)) * 0.8)
    selected_imgs = random.sample(os.listdir(image_dir), num_images)

    for image in selected_imgs:
        src = os.path.join(image_dir, image)
        dst = os.path.join(breed_train_dir, image)
        shutil.move(src, dst)
        print(f"Copied image: {image} from {src} to {dst}")

    for image in os.listdir(image_dir):
        src = os.path.join(image_dir, image)
        dst = os.path.join(val_breed_dir, image)
        shutil.move(src, dst)
        print(f"Copied image: {image} from {src} to {dst}")

os.system(f"del images")

retriever = WikipediaRetriever()
breeds = os.listdir("data/train")

context = []
for breed in breeds:
    context.extend([doc.page_content for doc in retriever.invoke(breed, top_k_results=1)])

with open("Breed_Data.txt", "w", encoding="utf-8") as file:
    for c in context:
        file.write(c + "\n")

os.makedirs(os.path.join("data", "text_data"), exist_ok=True)
shutil.move("Breed_Data.txt", "data/text_data")
