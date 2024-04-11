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
        # self.dense.add(tf.keras.layers.Dense(units=1024, activation='sigmoid'))
        # self.dense.add(tf.keras.layers.Dense(units=512, activation='sigmoid'))
        # self.dense.add(tf.keras.layers.Dense(units=128, activation='sigmoid'))
        # self.dense.add(tf.keras.layers.Dense(units=64, activation='sigmoid'))
        # self.dense.add(tf.keras.layers.Dropout(0.2))
        self.out = tf.keras.layers.Dense(units=num_action, activation='softmax')

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
        self.learning_rate = kwargs.get('learning_rate',0.1)
        self.batch_size = kwargs.get('batch_size',32)
        self.target_info_as_input = kwargs.get('target_info_as_input',False)
        self.save = kwargs.get('save',True)
        self.input_shape = input_shape
        self.label_num = num_action
        self.datas = datas
        self.labels = labels
        self._buf_count = 0
        self.train_loss = np.zeros((0))
        self.train_acc = np.zeros((0))
        self.loss = np.zeros((0))
        self.acc = np.zeros((0))
        
        self.net = CNNnet(num_action,input_shape,self.target_info_as_input)
        # self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        # self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.net.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
    
    def train_test_split(self,test_size=0.05):
        test_size = int(len(self.datas)*test_size)
        # np.random.seed(0)
        # rand_idx = np.random.permutation(len(self.datas))
        # self.train_datas = self.datas[rand_idx[test_size:]]
        # self.train_labels = self.labels[rand_idx[test_size:]]
        # self.test_datas = self.datas[rand_idx[:test_size]]
        # self.test_labels = self.labels[rand_idx[:test_size]]
        self.train_datas = tf.convert_to_tensor(self.datas[test_size:])
        self.train_labels = tf.convert_to_tensor(self.labels[test_size:])
        self.test_datas = tf.convert_to_tensor(self.datas[:test_size])
        self.test_labels = tf.convert_to_tensor(self.labels[:test_size])
    
    def learn(self,eopchs):
        best_loss = 100
        for epoch in range(eopchs):
            history=self.net.fit(self.train_datas,self.train_labels,batch_size=self.batch_size)
            train_loss = history.history['loss']
            train_acc = history.history['accuracy']
            self.train_loss = np.append(self.train_loss,train_loss)
            self.train_acc = np.append(self.train_acc,train_acc)
            loss,acc = self.net.evaluate(self.test_datas,self.test_labels)
            self.loss = np.append(self.loss,loss)
            self.acc = np.append(self.acc,acc)
            # pred = self.net.predict(self.test_datas)
            # pred = np.argmax(pred,axis=-1)
            # result_percent = np.zeros((self.label_num))
            # for i in range(self.label_num):
            #     result_percent[i] = np.sum(pred==i)
            # result_percent = result_percent/np.sum(result_percent)
            # print(result_percent)
            # print(f'epoch:{epoch} loss:{loss:.4f} acc:{acc:.4f} train_loss:{np.mean(loss_list):.4f} train_acc:{np.mean(acc_list):.4f}')
            if self.save and loss<best_loss:
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
        np.savetxt(path+'train_loss.txt',np.array(self.train_loss))
        np.savetxt(path+'train_acc.txt',np.array(self.train_acc))
        np.savetxt(path+'acc.txt',np.array(self.acc))
        np.savetxt(path+'loss.txt',np.array(self.loss))
    
    def load_model(self,path):
        # dummy input to build the model
        dummy_input = tf.zeros_like(self.test_datas)
        self.net(dummy_input)
        if not os.path.exists(path+'model.h5'):
            print('model not found')
            return
        self.net.load_weights(path+'model.h5')
        self.loss = np.loadtxt(path+'loss.txt')
        self.acc = np.loadtxt(path+'acc.txt')
        self.train_loss = np.loadtxt(path+'train_loss.txt')
        self.train_acc = np.loadtxt(path+'train_acc.txt')

def load_data():
    datas = np.load(data_path)
    datas = datas[1:,:2]
    datas = datas/200.0
    labels_crbmse = np.load(label_path)
    if len(labels_crbmse.shape)>1:
        labels_crbmse = labels_crbmse[1:,:]
        labels = np.zeros((labels_crbmse.shape[0]))
        labels_crb = labels_crbmse[:,:6]
        labels_mse = labels_crbmse[:,6:]
        for idx in range(labels_crbmse.shape[0]):
            min_crb_idx = np.argmin(labels_crb[idx])
            min_mse = np.min(labels_mse[idx])
            min_idx = np.where(labels_mse[idx]==min_mse)
            max_crb = np.argmax(labels_crb[idx,min_idx])
            labels[idx] = min_idx[0][max_crb]
        # labels[idx] = min_crb_idx
    else:
        labels = labels_crbmse[1:]
    # ---------------pirnt labels distribution----------------
    labels_num = np.zeros((6))
    for i in range(6):
        labels_num[i] = np.sum(labels==i)
    label_percent = labels_num/np.sum(labels_num)
    print(label_percent)
    # --------------------------------------------------------
    return datas,labels

def main():
    datas,labels = load_data()
    print(datas.shape)
    print(labels.shape)
    cnn = CNN(6,(6,6),datas,labels,target_info_as_input=True,learning_rate=2e-4,batch_size=512,save=True)
    cnn.train_test_split()
    cnn.load_model(model_save_path)
    cnn.learn(1000)
    cnn.save_model(model_save_path)

if __name__ == '__main__':
    data_path = './datas/datas-new.npy'
    label_path = './datas/labels-new.npy'
    model_save_path = 'model/0411/'
    main()