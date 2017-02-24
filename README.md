## Setup

Clone the repo and add the package to the python path.

```
git clone https://github.com/bradyz/geometry_processing.git
PYTHONPATH=$(pwd):$PYTHONPATH
```

## Data Dependencies

Modelnet with 25 viewpoints each - https://drive.google.com/open?id=0B0d9M5p2RxBqN0IzOXpudjMyTDQ

Our model's weights - https://drive.google.com/open?id=0B0d9M5p2RxBqMlNZOFg1YmlYR3c

## Contents

1. [View Generator](#view_gen) - take 2D projections of mesh files.
2. [Train CNN](#train_cnn) - fine tune a VGG-16 CNN on the new images.
3. [Classifier](#classify) - train a SVM on the CNN features.
3. [References](#references) - papers and resources used.

## View Generator <a name="view_gen"></a>

Given a model and a list of viewpoints - .png image files that correspond to 2D projection will be generated.

Preprocessing consists of centering the mesh, uniformly scaling the bounding box to a unit cube, and taking viewpoints that are centered at the centroid.

Currently there are 25 viewpoints being generated that fall around the unit sphere from 5 different phis and 5 different thetas (spherical coordinates).

## Train CNN <a name="train_cnn"></a>

The model used in this project is a VGG-16 with pretrained weights (ImageNet), with two additional layers fc1 (2048), fc2 (1024).

Training was done for 10 epochs on 100k training images (4000 meshes) over 10 labels of ModelNet10. The images were 224 x 224 rgb. Cross entropy loss was used in combination with a SGD optimizer with a batch size of 64. Training took approximately 5 hours.

After training, classification accuracy, given a single pose, is at 80% on a test set of 20k images.

## Classifier <a name="classify"></a>

The question asked is - given a mesh and several viewpoints, does it help to use all of the viewpoints (MVCNN), or does a selected subset of size k give better accuracy?

We use a one-vs-rest linear SVM, similar to MVCNN, to classify activation values of the final fc layer.

The current methods consist of the using the following(currently unimplemented) -

* Sort by minimized entropy

## References <a name="references"></a>

Multi-view Convolutional Neural Networks for 3D Shape Recognition - https://arxiv.org/pdf/1505.00880.pdf
Princeton ModelNet - http://modelnet.cs.princeton.edu/
