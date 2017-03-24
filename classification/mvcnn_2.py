import os
import cv2
import sys
import numpy as np
from fnmatch import fnmatch
import keras.layers
from keras.models import Model, Sequential
from sklearn import linear_model
from matplotlib import pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

sys.path.insert(0, '/home/04005/jl57472/keras_tf')
from agp_proj.utils.helpers import (TRAIN_DIR, VALID_DIR,
        SAVE_FILE, LOG_FILE, IMAGE_SIZE, NUM_CLASSES, get_data)
from agp_proj.train_cnn.classify_keras import load_model_vgg

BATCH_SIZE = 16
BATCH = 64

def image_from_path(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    try:
        image = np.reshape(image, (1, 224, 224, 3))
    except:
        print('IMAGE LOADING FAILDED!!!')
        print('->path: %s' % image_path)
        raise

    return image


def output_entropy(prediction, eps):
    return np.sum(-(prediction+eps)*np.log2(prediction+eps))

# Image must be of size (1, 224, 224, 3).
def test_from_image(model, image_path):
    image = image_from_path(image_path)
    prediction = model.predict(image)
    class_name = np.sort(os.listdir(VALID_DIR))[np.argmax(prediction, axis=1)]
    return prediction, class_name

def feature_extraction(model, featurelayer, imagepath):
    image = image_from_path(imagepath)
    intermediate_layer_model = Model(input=model.input,
                                     output=model.get_layer(featurelayer).output)
    feature_output = intermediate_layer_model.predict(image)

    return feature_output

def fps_selection(model,
                  img_dir, 
                  objname,
                  modelid,
                  featurelayer,
                  numselection,
                  ):
    files_pattern="*"+modelid+".off_*.png"
    viewimgs = gfile.Glob(VALID_DIR+objname+"/"+files_pattern)
    feats = filenames = np.array([])

    for file in np.sort(viewimgs):
        sys.stdout.write('.')
        feat = feature_extraction(model, featurelayer, file)
        filenames = np.append(filenames, file)
        feats = np.append(feats, feat)
    feats = np.reshape(feats,
                       (filenames.shape[0], model.get_layer(featurelayer).output.shape[1]))
    
    solution_feats = solution_filenames = []
    feats = feats.tolist()
    filenames = filenames.tolist()
    initindx = rd.randint(0, len(filenames)-1)
    solution_feats = np.append(solution_feats, feats.pop(initindx))
    solution_filenames = np.append(solution_filenames, filenames.pop(initindx))
    
    for _ in range(numselection-1):
        distances = [feat_distance(f, solution_feats[0]) for f in feats]
        for i, f in enumerate(feats):
            for j, s in enumerate(solution_feats):
                distances[i] = min(distances[i], feat_distance(f, s))
        solution_feats = np.append(solution_feats, feats.pop(distances.index(max(distances))))
        solution_filenames = np.append(solution_filenames, filenames.pop(distances.index(max(distances))))
    solution_feats = np.asarray(solution_feats)
    solution_feats = np.reshape(solution_feats,
                               (len(solution_filenames), model.get_layer(featurelayer).output.shape[1]))
    solution_filenames = np.asarray(solution_filenames)
    sys.stdout.write('!\n')
    return solution_feats, solution_filenames

def feature_pooling(model,
                    img_dir,
                    bestviews,
                    featurelayer):

    img_dir += objname + '/'
    feature_mv = np.array([])
    numfiles = 0

    for file in bestviews:
        if os.path.isfile(img_dir+file):
            #print file
            sys.stdout.write('.')
            feature_output = feature_extraction(model, featurelayer, img_dir+file)
            feature_mv = np.append(feature_mv, feature_output[0])
            numfiles+=1
    if numfiles>1:
        feature_mv = np.reshape(feature_mv,
                                (numfiles, model.get_layer(featurelayer).output.shape[1]))
        feature_mv = np.amax(feature_mv, axis=0)
    return feature_mv

def view_selection(model,
                   img_dir,
                   objname,
                   modelid,
                   num_selection,
                   eps):

    img_dir += objname + '/'
    filename = objname + '_' + modelid +'.off_'
    numfiles = 0
    entropy = filenames = bestfilenames = bestentropy = np.array([])
    filelist = os.listdir(img_dir)

    for file in np.sort(filelist):
        if fnmatch(file, '*'+modelid+'*'):
            sys.stdout.write('.')
            prediction, label = test_from_image(model, img_dir+file)
            #print isinstance(prediction, np.ndarray) # check if it is array instance
            entropy = np.append(entropy, output_entropy(prediction.astype(np.float128), np.float128(eps)))
            filenames = np.append(filenames, file)
            numfiles += 1
    sortindx = np.argsort(entropy)
    for i in range(0,num_selection):
        bestentropy = np.append(bestentropy, entropy[sortindx[-i-1]])
        bestfilenames = np.append(bestfilenames, filenames[sortindx[-i-1]])
    sys.stdout.write('!\n')
    return bestfilenames, bestentropy

def mvcnn(model,
          img_dir,
          bestviews,
          featurelayer):

    agg_feature = feature_pooling(model, img_dir, bestviews, featurelayer)

    feat_input = Input(tensor=Input(shape=(agg_feature.shape)))
    x = model.get_layer('fc2')(feat_input)
    x = model.get_layer('predictions')(x)
    cnn2_model = Model(input=feat_input, output=x)

    prediction = cnn2_model.predict(np.array([agg_feature]))
    class_name = np.sort(os.listdir(VALID_DIR))[np.argmax(prediction, axis=1)]
    sys.stdout.write('!\n')
    return prediction, class_name


if __name__ == '__main__':

    vggmodel = load_model_vgg()

    layername = 'fc1'
    objname = 'bed'
    objID = '0517.off_'
    numselection = 7

    testImg_path = VALID_DIR+'bed/'+'bed_0517.off_1_1.png'
    output_probs, output_label = test_from_image(vggmodel, testImg_path)
    print output_label

    bestviews, bestentr = view_selection(vggmodel,
                                       VALID_DIR,
                                       objname,
                                       objID,
                                       numselection,
                                       eps=10e-4 )
    prediction, class_name = mvcnn(vggmodel,
                                   VALID_DIR,
                                   bestviews,
                                   layername)
    print bestviews
    print bestentr
    print class_name
    print('Done.')
                                                             
