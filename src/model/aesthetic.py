import cv2
import numpy as np
from keras import optimizers
from keras.layers import Dense,Activation,Input
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.merge import concatenate,add
from keras.models import Model

from keras.applications.resnet50 import preprocess_input
import keras.backend as K

import resnet50_model as custom_resnet50


class model:
    def __init__(self,input_size=(3,224,224),output_shape=2,dense_shape=2048):
        self.input_size = input_size
        self.model = custom_resnet50.ResNet50(include_top=False,input_shape=input_size)
        self.__load_trained_weights(output_shape=2,dense_shape=2048)

    def __load_trained_weights(self,dense_shape=2048,output_shape=2):
        x = self.model.output
        x = Flatten()(x)
        x = Dense(dense_shape, activation='relu')(x)
        x = Dropout(0.5)(x)
        prediction = Dense(output_shape, activation='softmax')(x)

        self.model = Model(inputs=self.model.input, outputs=prediction)
        self.model.load_weights('../weights/sync_resnet50_binary.h5')
        #print (self.model.summary())

    def modify_last_layers(self,dense_shape,output_shape):
        for i in range(3):
            self.model.layers.pop()
        x = self.model.layers[-1].output
        x = Dense(dense_shape, activation='relu')(x)
        x = Dropout(0.5)(x)
        prediction = Dense(output_shape, activation='softmax')(x)
        self.model = Model(inputs=self.model.input, outputs=prediction)
        #print self.model.summary()

    def predict(self):
        self.model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9))
        pred_all = []
        pred_pr_all = []
        for x_te,y_te in image_iterator_binary_x_only(te_list,batch_size=1):
            pred = model.predict_on_batch(x_te)
            pred_all.append(np.argmin(pred,axis=1))
            pred_pr_all.append(pred[:,0])
            #print '------------------------'
        pred_all = np.asarray(pred_all).flatten(order='C')
        pred_pr_all = np.asarray(pred_pr_all).flatten(order='C')
        #all_name = [x[0] for x in te_list]
        all_name = te_list

    def batch_predict(self,input_data):

        pred = self.model.predict_on_batch(input_data)

        pred_class = np.argmin(pred,axis=1)
        pred_proba = pred[:,0]

        return pred_class,pred_proba
    def batch_predict_raw(self,input_data):
        #pred_class = []
        #pred_proba = []

        pred = self.model.predict_on_batch(input_data)
        return pred

    def batch_train(self,X,Y):
        batch_loss = self.model.train_on_batch(X, Y)
        return batch_loss
    def batch_test(self,X,Y):
        batch_loss = self.model.test_on_batch(X, Y)
        return batch_loss
    def compile(self,lr=1e-4):
        self.model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=lr, momentum=0.9))

def image_cv_reader(image_path, img_size=None):
    img = read_image(image_path, img_size)
    x = np.expand_dims(img, axis=0)
    x = x.astype(np.float64)
    x = preprocess_input(x)
    #print x.shape,'!!!!'
    return x[0,:,:,:]

def read_image(img_path,size=None):
    img = cv2.imread(img_path)
    if size is not None:
        img = cv2.resize(img,size)
    return convert_image_format_BGR2RGB_channellast2first(img)

def convert_image_format_BGR2RGB_channellast2first(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    #channel first
    img = img.transpose(2, 0, 1)
    return img











if __name__ == "__main__":
    #test_imgread()
    a = read_image('/home/demcare/Github/mtcnn/01C75A41-0321-4929-96B6-7BCFB56D3D40.jpg',(224,224))
    a = np.asarray([a,a,a])
    print a.shape
    model = model()
    #model.modify_last_layers(4,1024)
    print model.batch_predict(a)
