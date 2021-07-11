import os
import numpy as np
from helper import get_spectograms
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import pickle
import argparse

def main(args):

    model_path = args.model_path

    classifier = load_model(model_path+"classifier_weights.hdf5")
    lb = pickle.loads(open(model_path+"classifier_classes.lbl", "rb").read())
    scale_vals = np.load(model_path+"scale_vals.npy")
    fps = 24
    TIME_CHUNK = 3
    X_test, Y_test = [], []
    max_dopVal = scale_vals[0]
    max_synth_dopVal =  scale_vals[1]
    min_dopVal = scale_vals[2]
    min_synth_dopVal = scale_vals[3]

    dop_file = args.video_path + 'output/video' + '/synth_doppler.npy'
    # dop_file = args.video_path + 'doppler_gt.npy'
    dopler = np.load(dop_file)
    dopler = get_spectograms(dopler, TIME_CHUNK, fps, synthetic=True, zero_pad=True)
    class_arr = np.array([args.action_tobe] * dopler.shape[0])
    dopler = dopler.astype("float32")
    dopler = (dopler - min_dopVal)/(max_dopVal - min_dopVal)
    X_test.append(dopler)
    Y_test.append(class_arr)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)


    Y_pred, Y_gt = [], []
    # for i in range(X_test.shape[0]):
    X_in = np.expand_dims(X_test[0], axis=-1)
    class_lbl = Y_test[0][0]
    proba = np.mean(classifier.predict(X_in),axis=0)
    print(proba)
    idx = np.argmax(proba)
    label = lb.classes_[idx]
    print(idx)
    print(label)
    Y_pred.append(label)
    Y_gt.append(class_lbl)

    acc = np.round(100*accuracy_score(Y_gt, Y_pred),2)
    print("Train on Synthetic Only & Test on Real World Doppler - Accuracy:",acc,"%")


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--model_path', type=str, help='Path to DL models')

	parser.add_argument('--video_path', type=str, help='Path to')

	parser.add_argument('--action_tobe', type=str, help='Input the action to be detected')

	args = parser.parse_args()

	main(args)

