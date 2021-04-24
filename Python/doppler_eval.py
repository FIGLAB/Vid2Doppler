import os
import numpy as np
from helper import get_spectograms
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import pickle

path = "/Users/karanahuja/Dropbox/DeployedData/Vid2Doppler/"
data_path = path + "data/"
model_path = path + "models/"

classifier = load_model(model_path+"classifier_weights.hdf5")
lb = pickle.loads(open(model_path+"classifier_classes.lbl", "rb").read())
scale_vals = np.load(model_path+"scale_vals.npy")
classes = ['Waving', 'WalkingUpSteps', 'Walking', 'Squat', 'Running', 'Lunge', 'JumpRope', 'JumpingJack', 'Jumping', 'Cycling', 'Cleaning', 'Clapping']
participants = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10']
angles = ['angle_0','angle_45','angle_minus_45']
fps = 24
TIME_CHUNK = 3
X_test, Y_test = [], []
max_dopVal = scale_vals[0]
max_synth_dopVal =  scale_vals[1]
min_dopVal = scale_vals[2]
min_synth_dopVal = scale_vals[3]

for participant in participants:
    for angle in angles:
        in_path = data_path + '/' + participant + '/' + angle + '/'
        for folder in  classes:
            path = in_path + '/' + folder + '/'
            if os.path.isdir(path):
                subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]
                for s_folder in subfolders:
                    class_name = s_folder.split("/")[-1].split("_")[0]
                    instance_id = int(s_folder.split("/")[-1].split("_")[1])
                    dop_file = s_folder + '/doppler_gt.npy'
                    dopler = np.load(dop_file)
                    dopler = get_spectograms(dopler, TIME_CHUNK, fps)
                    class_arr = np.array([class_name] * dopler.shape[0])
                    dopler = dopler.astype("float32")
                    dopler = (dopler - min_dopVal)/(max_dopVal - min_dopVal)
                    X_test.append(dopler)
                    Y_test.append(class_arr)

X_test = np.array(X_test)
Y_test = np.array(Y_test)


Y_pred, Y_gt = [], []

for i in range(X_test.shape[0]):
    X_in = np.expand_dims(X_test[i], axis=-1)
    class_lbl = Y_test[i][0]
    proba = np.mean(classifier.predict(X_in),axis=0)
    idx = np.argmax(proba)
    label = lb.classes_[idx]
    Y_pred.append(label)
    Y_gt.append(class_lbl)

acc = np.round(100*accuracy_score(Y_gt, Y_pred),2)
print("Train on Synthetic Only & Test on Real World Doppler - Accuracy:",acc,"%")
