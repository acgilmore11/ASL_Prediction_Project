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


word_path = dir_path + "\\" + "word-videos" #Directory that stores word videos

crop_word_path = dir_path + "\\" + "Cropped_Word_Images\\" #Directory that stores the cropped images from word videos
ann_word_path = dir_path + "\\annotated_Word_Images\\" #directory that stores the annotated images from word videos



w_path = glob.glob(word_path + "\\" + "*.mp4")

if (os.path.exists(crop_word_path) == False):
    os.mkdir(crop_word_path)
if (os.path.exists(ann_word_path) == False):
    os.mkdir(ann_word_path)

##############################################################################################################

model = hfe.HandShapeFeatureExtractor.get_instance()

#hand_extraction on each letter in word + CNN prediction
for word_paths in w_path:
    w_name = Path(word_paths).stem
    frames = heu.vid_segment(word_paths)
    letterIndex = 0
    predicted = ""
    for f in frames:
        crop_img,ann_img = heu.hand_extraction_on_frame(f)
        crop_img_word_path = crop_word_path + w_name + str(letterIndex) + '.jpg'
        ann_image_word_path = ann_word_path + w_name + str(letterIndex) + '.jpg'
        cv2.imwrite(crop_img_word_path, crop_img[0])
        cv2.imwrite(ann_image_word_path, cv2.flip(ann_img, 1))

        img = cv2.imread(crop_img_word_path)
        #img = cv2.rotate(img, cv2.ROTATE_180)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = model.extract_feature(img)

        pred_letter = ota.ota_dict[np.argmax(res)]
        predicted += pred_letter

        letterIndex = letterIndex + 1
    print("Actual word: " + w_name + ", Predicted: " + predicted)




