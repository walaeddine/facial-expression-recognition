# emotion-recognition


## [Facial Expression Recognition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

train.csv contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image. The contents of this string a space-separated pixel values in row major order. test.csv contains only the "pixels" column and your task is to predict the emotion column.

The training set consists of 28,709 examples. The public test set used for the leaderboard consists of 3,589 examples. The final test set, which was used to determine the winner of the competition, consists of another 3,589 examples.

This dataset was prepared by Pierre-Luc Carrier and Aaron Courville, as part of an ongoing research project. They have graciously provided the workshop organizers with a preliminary version of their dataset to use for this contest.


##### Project structure

```
.
├── checkpoints [28 entries exceeds filelimit, not opening dir]
├── config
│   ├── emotion_config.py
│   └── __init__.py
├── datasets
│   ├── fer2013
│   │   └── fer2013.csv
│   └── hdf5
│       ├── test.hdf5
│       ├── train.hdf5
│       └── val.hdf5
├── output
│   ├── vggnet_emotion.json
│   └── vggnet_emotion.png
├── pipeline
│   ├── callbacks
│   │   ├── epochcheckpoint.py
│   │   ├── __init__.py
│   │   └── trainingmonitor.py
│   ├── conv
│   │   ├── emotionvggnet.py
│   │   └── __init__.py
│   ├── io
│   │   ├── hdf5datasetgenerator.py
│   │   ├── hdf5datasetwriter.py
│   │   └── __init__.py
│   ├── preprocessing
│   │   ├── imagetoarraypreprocessor.py
│   │   └── __init__.py
│   └── __init__.py
├── build_dataset.py
├── emotion_detector.py
├── haarcascade_frontalface_default.xml
├── README.md
├── test_recognizer.py
└── train_recognizer.py
```

Clone the repository:
```
git clone git@github.com:walaeddine/emotion-recognition.git
```

Download the dataset [FER2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

create a dataset folders
```
mkdir datasets
mkdir datasets/fer2013
mkdir datasets/hdf5
```

Unzip the dataset file into datasets/fer2013
```
unzip fer2013-dataset.zip
```

Go ahead and create the training, testing, and validation split directory structure by executing the following command:

```
python build_dataset.py
```

Go ahead and train emovggnet on our fer2013 dataset for 100 epochs and lr 1e-3

```
python train_recognizer.py --checkpoints checkpoints
```

Train for another 20 epochs with lr 1e-4
```
python train_recognizer.py --checkpoints checkpoints/epoch_100.hdf5 --start-epoch 101
```

train the model for another 20 epochs with lr 1e-5
```
python train_recognizer.py --checkpoints checkpoints/epoch_120.hdf5 --start-epoch 121
```


Test the model
```
python test_recognizer.py
```

Results:
```
[INFO] evaluating network...

acc: 0.68
```
![Accuracy/Loss Plot](https://github.com/walaeddine/facial-expression-recognition/blob/main/output/vggnet_emotion.png?raw=true)
