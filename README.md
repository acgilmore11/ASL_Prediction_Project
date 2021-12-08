# ASL_Prediction_Project

To test word prediction:
1. Record ASL word video. Make sure each letter is 3 seconds long and that camera is recording at 30 fps.
2. Name file as signed word (ex. dog.mp4)
3. Insert file in "word-videos" folder. This folder already contains our group's word videos.
4. Run main.py. "annotated_Word_Images" and "Cropped_Word_Images" folders will be generated and populated 
upon execution.

To test CNN accuracy (generate F1-Scores)
1. Run cnn_accuracy_test.py. This will generate the F1 scores of each class based on the alphabet videos in 
the "Alphabet-videos" folders. "annoted_images", "Cropped_Images", and "middle_frames" folders will be generated
and populated upon execution.


File Descriptions:
flask-backend.zip --> server that receives recorded videos
AlphabetVideoUploader.zip --> Android app for uploading alphabet videos
WordVideoUploader.zip --> Android app for uploading word videos
hand_extractor_updated.py --> python script to crop hand from frame
f1_score.py --> calculates f1 scores for each classification
handshape_feature_extractor.py --> processes image and creates feature vector using cnn model
output_to_alphabet.py --> contains dict that converts class index to letter
cnn_accuracy_test --> driver script that generates f1 scores from alphabet videos
main.py --> driver script that generates predicted words from input videos


