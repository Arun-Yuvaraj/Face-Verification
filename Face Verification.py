#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import Packages
from matplotlib import pyplot
import cv2
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from PIL import Image
from numpy import asarray
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from scipy.spatial.distance import cosin
from numpy import expand_dims


# In[2]:


# etract face of different persons
def extract_face(filename, required_size=(224, 224)): 
    pixels = pyplot.imread(filename) 
    detector = MTCNN()
    results = detector.detect_faces(pixels) 
    x1, y1, width, height = results[0]['box'] 
    x2, y2 = x1 + width, y1 + height 
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face) 
    image = image.resize(required_size) 
    face_array = asarray(image) 
    return face_array


# In[3]:


# get embeddings from last layer of Resnet which can be used for Verification
def get_embeddings(filenames): 
    faces = [extract_face(f) for f in filenames] 
    samples = asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2) 
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg') 
    yhat = model.predict(samples) 
    return yhat


# In[9]:


def is_match(known_embedding, candidate_embedding, thresh=0.6): 
    score = cosine(known_embedding, candidate_embedding) 
    if score <= thresh: 
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh)) 
    else: 
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))


# In[5]:


filenames = ['john.jfif','steve_1.jfif','steve_2.jfif']


# In[6]:


embeddings = get_embeddings(filenames) 
steve = embeddings[1]


# In[10]:


# is_match function can be used to compare distance between embeddings of faces
is_match(embeddings[1], embeddings[2])
is_match(embeddings[1], embeddings[0])

