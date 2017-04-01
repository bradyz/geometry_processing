import pickle
import os
import cv2
import sys
import Image
import numpy as np
import random as rd
from fnmatch import fnmatch
from sklearn import linear_model
from matplotlib import pyplot as plt
%matplotlib inline

import keras.layers
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Input
from tensorflow.python.platform import gfile
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

SAVE_WEIGHTS_FILE = '/your/weights/path/model_weights.h5'
VALID_DIR = "/your/testimages/path/ModelNetViewpoints/test/"
IMAGE_SIZE = 224
NUM_CLASSES = 10

def load_model_vgg():
    img_input = Input(tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    base_model = VGG16(include_top=False, input_tensor=img_input)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dense(2048, activation='relu', name='fc1')(x)
    x = Dense(1024, activation='relu', name='fc2')(x)
    x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
    model = Model(input=img_input, output=x)
    model.load_weights(SAVE_WEIGHTS_FILE, by_name=True)
    #print('Model loaded with weights from %s.' % SAVE_WEIGHTS_FILE)
    
    return model

class mvcnnclass:
    
    def __init__(self,
                 model,
                 featurelayer='fc1',
                 numallview=25,
                 numviewselection=7,
                 data_path=VALID_DIR
                ):
        self.model = model
        self.featurelayer = featurelayer
        self.numallview = numallview
        self.numviewselection = numviewselection
        self.data_path = data_path
    
    def get_data(self, objname):
        viewimgfilepaths, objIDs = [],[]
        objlist = np.sort(os.listdir(self.data_path+objname+'/'))
        modelnum = objlist[::self.numallview]
        for i,name in enumerate(modelnum):
            name = name.replace('.off_1_1.png','')
            files = np.sort(gfile.Glob(self.data_path+objname+'/'+name+'*'))
            viewimgfilepaths.append(files)
            objIDs.append(name)
        print 'Views are loaded!'
        return viewimgfilepaths, objIDs
    
    def image_from_path(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        try:
            image = np.reshape(image, (1, IMAGE_SIZE, IMAGE_SIZE, 3)) 
        except:    
            print('IMAGE LOADING FAILDED!!!')
            print('->path: %s' % image_path)
            raise
        return image

    def singleview_classification(self, image_path):
        image = self.image_from_path(image_path)
        prediction = self.model.predict(image)
        class_name = np.sort(os.listdir(VALID_DIR))[np.argmax(prediction, axis=1)]
        return prediction, class_name
    
    def feat_distance(self, feat1, feat2):
        sim = spatial.distance.cosine(feat1, feat2)
        return 1-sim 
    
    def output_entropy(self, prediction, eps=10e-4):
        return np.sum(-(prediction+eps)*np.log2(prediction+eps))

    def feature_extraction(self, imagepath):
        image = self.image_from_path(imagepath)
        intermediate_layer_model = Model(input=self.model.input,
                                         output=self.model.get_layer(self.featurelayer).output)
        return intermediate_layer_model.predict(image)

    def entropy_selection(self, viewimgfilepaths):
        filenames = []
        entropies = []
        for file in viewimgfilepaths:
            sys.stdout.write('.')
            prediction, _ = self.singleview_classification(file)
            entropies = np.append(entropies, self.output_entropy(prediction))
            filenames = np.append(filenames, file)
        solution_feats, solution_filepath = [],[]
        argminentropy = np.argsort(entropies)[:self.numviewselection]
        solution_filepath = filenames[argminentropy]
        for file in solution_filepath:
            solution_feats.append(self.feature_extraction(file))
        solution_feats = np.asarray(solution_feats)
        solution_feats = np.reshape(solution_feats,
                                    (solution_filepath.shape[0],
                                     self.model.get_layer(self.featurelayer).output.shape[1]))
        solution_filepath = np.asarray(solution_filepath)
        return solution_feats, solution_filepath
    
    def fps_selection(self, viewimgfilepaths):
        feats, filenames = [],[]
        for file in viewimgfilepaths:
            sys.stdout.write('.')
            feat = self.feature_extraction(file)
            filenames.append(file)
            feats.append(feat)
        solution_feats, solution_filepath = [],[]
        initindx = rd.randint(0, len(filenames)-1)
        
        solution_feats.append(feats.pop(initindx))
        solution_filepath.append(filenames.pop(initindx))
        
        for i in range(self.numviewselection-1):
            distances = [self.feat_distance(f, solution_feats[0]) for f in feats]
            for i, f in enumerate(feats):
                for j, s in enumerate(solution_feats):
                    distances[i] = min(distances[i], self.feat_distance(f, s))
            solution_feats.append(feats.pop(distances.index(max(distances))))
            solution_filepath.append(filenames.pop(distances.index(max(distances))))
        solution_feats = np.asarray(solution_feats)
        solution_feats = np.reshape(solution_feats,
                                    (len(solution_filepath), 
                                     self.model.get_layer(self.featurelayer).output.shape[1]))
        solution_filepath = np.asarray(solution_filepath)
        sys.stdout.write('!\n')
        print "FPS selection done."
        return solution_feats, solution_filepath
    
    def feature_pooling(self, selected_feats):
        return np.amax(selected_feats, axis=0)

    def mvcnn_classification(self, objname):
        predictions, class_names = [],[]
        viewimgfilepaths, objID =mvcnn.get_data(objname)
        for i, objIDpaths in enumerate(viewimgfilepaths):
            print "Object ID-> %s" % objID[i]
            bestfeats, bestfilepath = self.entropy_selection(objIDpaths)
            agg_feat = self.feature_pooling(bestfeats)

            feat_input = Input(tensor=Input(shape=(agg_feat.shape)))
            if self.featurelayer=='fc1':
                x = self.model.get_layer('fc2')(feat_input)
                x = self.model.get_layer('predictions')(x)
            else:
                x = self.model.get_layer('predictions')(feat_input)
            cnn2_model = Model(input=feat_input, output=x)

            prediction = cnn2_model.predict(np.array([agg_feat]))
            class_name = np.sort(os.listdir(VALID_DIR))[np.argmax(prediction, axis=1)]
            sys.stdout.write('!\n')
            predictions.append(prediction)
            class_names.append(class_name)
        print "Classification Done."
        return prediction, class_name

if __name__ == '__main__':
    vggmodel = load_model_vgg()
    mvcnn = mvcnnclass(vggmodel, featurelayer='fc1')
    pred, classname = mvcnn.mvcnn_classification('bed')
