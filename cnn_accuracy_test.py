import hand_extractor_updated as heu
import handshape_feature_extractor as hfe
import os
import glob
import cv2
import numpy as np
from pathlib import Path
import f1_score as f1
import output_to_alphabet as ota

dir_path = os.path.dirname(os.path.realpath(__file__))
v_d_path = dir_path + "\\" + "Alphabet-videos" #Directory that stores the collected videos
v_d_path1 = dir_path + "\\" + "Alphabet-videos1" #Directory that stores the collected videos
v_d_path2 = dir_path + "\\" + "Alphabet-videos2" #Directory that stores the collected videos
v_d_path3 = dir_path + "\\" + "Alphabet-videos3" #Directory that stores the collected videos

crop_path = dir_path + "\\" + "Cropped_Images\\" #Directory that stores the cropped images
ann_path = dir_path + "\\annotated_Images\\" #directory that stores the annotated images
mid_path = dir_path + "\\middle_frames\\" #directory that stores the middle images
asl_path = dir_path + "\\asl_alphabet_test\\" #directory that stores the test images from the asl dataset from kaggle

v_path = glob.glob(v_d_path + "\\" + "*.mp4")
v_path1 = glob.glob(v_d_path1 + "\\" + "*.mp4")
v_path2 = glob.glob(v_d_path2 + "\\" + "*.mp4")
v_path3 = glob.glob(v_d_path3 + "\\" + "*.mp4")

alpha_paths_list = []
alpha_paths_list.append(v_path)
alpha_paths_list.append(v_path1)
alpha_paths_list.append(v_path2)
alpha_paths_list.append(v_path3)


if (os.path.exists(crop_path) == False):
    os.mkdir(crop_path)
if (os.path.exists(ann_path) == False):
    os.mkdir(ann_path)
if (os.path.exists(mid_path) == False):
    os.mkdir(mid_path)

for p in alpha_paths_list:
    for vid_paths in p:
        v_name = Path(vid_paths).stem
        crop_img,ann_img,mid_frame = heu.hand_extraction(vid_paths)
        crop_img_path = crop_path + v_name + '.jpg'
        ann_image_path = ann_path + v_name + '.jpg'
        mid_frame_path = mid_path + v_name + '.jpg'
        cv2.imwrite(crop_img_path, crop_img[0])
        cv2.imwrite(ann_image_path, cv2.flip(ann_img, 1))
        cv2.imwrite(mid_frame_path, cv2.flip(mid_frame, 1))

model = hfe.HandShapeFeatureExtractor.get_instance()

img_vectors = {}
img_vectors1 = {}
img_vectors2 = {}
img_vectors3 = {}
test_vectors = {}

crop_image_path = glob.glob(crop_path + "*.jpg")
for img_path in crop_image_path:
    i_name = Path(img_path).stem
    img = cv2.imread(img_path)
    img = cv2.rotate(img, cv2.ROTATE_180)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = model.extract_feature(img)
    if '1' in i_name:
        img_vectors1[i_name] = res
    elif '2' in i_name:
        img_vectors2[i_name] = res
    elif '3' in i_name:
        img_vectors3[i_name] = res
    else:
        img_vectors[i_name] = res

#results of austin alphabet videos
results = []

#results of manoj alphabet videos
results1 = []

#results of neeraj alphabet videos
results2 = []

#results of anant alphabet videos
results3 = []

for key in img_vectors:
    results.append(np.argmax(img_vectors[key]))

for key in img_vectors1:
    results1.append(np.argmax(img_vectors1[key]))

for key in img_vectors2:
    results2.append(np.argmax(img_vectors2[key]))

for key in img_vectors3:
    results3.append(np.argmax(img_vectors3[key]))


lst = []
lst.append(results)
lst.append(results1)
lst.append(results2)
lst.append(results3)
scores = f1.calculate_f1_scores(lst)


char = "a"
j = ord(char[0])
i = 0
print("F-Scores: ")
lim = j + 26
while j < lim:
    print("Class '" + chr(j) + "': " + str(scores[i]))
    i += 1
    j += 1