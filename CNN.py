import os
import time
import tqdm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class CNNnet(tf.keras.Model):
    def __init__(self, num_action,input_shape,target_info_as_input=False):
        super().__init__()
        self.target_info_as_input = target_info_as_input
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu',input_shape=input_shape)
        # self.maxpool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')
        self.flat = tf.keras.layers.Flatten()
        self.dense = tf.keras.Sequential()
        self.dense.add(tf.keras.layers.Dense(units=64, activation='relu'))
        self.dense.add(tf.keras.layers.Dense(units=128, activation='relu'))
        self.dense.add(tf.keras.layers.Dense(units=512, activation='relu'))
        self.dense.add(tf.keras.layers.Dense(units=1024, activation='relu'))
        self.dense.add(tf.keras.layers.Dense(units=512, activation='relu'))
        self.dense.add(tf.keras.layers.Dense(units=128, activation='relu'))
        self.dense.add(tf.keras.layers.Dense(units=64, activation='relu'))
        self.out = tf.keras.layers.Dense(units=num_action,activation='linear')

    def call(self, inputs):
        x = inputs
        if not self.target_info_as_input:
            x = self.conv1(inputs)
            # x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flat(x)
        x = self.dense(x)
        return self.out(x)

class CNN():
    def __init__(self,num_action,input_shape,datas,labels, **kwargs):
        self.learning_rate = kwargs.get('learning_rate',0.01)
        self.batch_size = kwargs.get('batch_size',32)
        self.target_info_as_input = kwargs.get('target_info_as_input',False)
        self.input_shape = input_shape
        self.datas = datas
        self.labels = labels
        self._buf_count = 0
        self.train_loss = []
        self.loss=[]
        self.acc=[]
        
        self.net = CNNnet(num_action,input_shape,self.target_info_as_input)
        self.net.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    
    def train_test_split(self,test_size=0.1):
        test_size = int(len(self.datas)*test_size)
        randidx = np.random.permutation(len(self.datas))
        self.train_datas = self.datas[randidx[test_size:]]
        self.train_labels = self.labels[randidx[test_size:]]
        self.test_datas = self.datas[randidx[:test_size]]
        self.test_labels = self.labels[randidx[:test_size]]
    
    def learn(self,eopchs):
        best_loss = 100
        for epoch in range(eopchs):
            bar = tqdm.tqdm(range(0,len(self.train_datas),self.batch_size))
            randidx = np.random.permutation(len(self.train_datas))
            for i in range(0,len(self.train_datas),self.batch_size):
                if i+self.batch_size>len(self.train_datas):
                    batch_datas = self.train_datas[randidx[i:]]
                    batch_labels = self.train_labels[randidx[i:]]
                batch_datas = self.train_datas[randidx[i:i+self.batch_size]]
                batch_labels = self.train_labels[randidx[i:i+self.batch_size]]
                self.net.train_on_batch(batch_datas,batch_labels)
                bar.update(1)
            loss,acc = self.net.evaluate(self.train_datas,self.train_labels)
            self.loss.append(loss)
            self.acc.append(acc)
            print('epoch:',epoch,'loss:',loss,'acc:',acc)
            if loss<best_loss:
                best_loss = loss
                self.save_model(model_save_path)
                print('==================model saved!==================')
            
    def predict(self,data):
        data_real = tf.math.real(data)
        data_img = tf.math.imag(data)
        data = tf.concat([data_real,data_img],axis=-1)
        data = tf.expand_dims(data,axis=0)
        data = tf.convert_to_tensor(data)
        pred = self.net.predict(data)[0]
        return tf.argmax(pred)
    
    def save_model(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.net.save_weights(path+'model.h5')
        with open(path+'loss.txt','w') as f:
            f.write(str(self.loss))
        with open(path+'acc.txt','w') as f:
            f.write(str(self.acc))
    
    def load_model(self,path):
        if not os.path.exists(path+'model.h5'):
            print('model not found')
            return
        self.net.load_weights(path+'model.h5')
        with open(path+'loss.txt','r') as f:
            self.loss = eval(f.read())
        with open(path+'acc.txt','r') as f:
            self.acc = eval(f.read())

def load_data(path):
    datas = np.load(path+'datas.npy')
    datas = datas[:,:3]
    labels = np.load(path+'labels.npy')
    return datas,labels

def main():
    model_save_path = 'model/only_pos_as_input/'
    datas,labels = load_data('')
    print(datas.shape)
    print(labels.shape)
    cnn = CNN(6,(6,6),datas,labels,target_info_as_input=True,learning_rate=0.001,batch_size=32)
    # cnn.load_model('model/')
    cnn.train_test_split()
    cnn.learn(500)
    cnn.save_model(model_save_path)

if __name__ == '__main__':
    model_save_path = 'model/only_pos_as_input/'
    main()