import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class U_Net():
    def __init__(self):
        # 设置图片基本参数
        self.height = 256
        self.width = 256
        self.channels = 1
        self.shape = (self.height, self.width, self.channels)

        # 优化器
        optimizer = Adam(0.002, 0.5)

        # u_net
        self.unet = self.build_unet()  # 创建网络变量
        self.unet.compile(loss='mse',
                          optimizer=optimizer,
                          metrics=[self.metric_fun])
        self.unet.summary()

    def build_unet(self, n_filters=16, dropout=0.1, batchnorm=True, padding='same'):

        # 定义一个多次使用的卷积块
        def conv2d_block(input_tensor, n_filters=16, kernel_size=3, batchnorm=True, padding='same'):
            # the first layer
            x = Conv2D(n_filters, kernel_size, padding=padding)(
                input_tensor)
            if batchnorm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # the second layer
            x = Conv2D(n_filters, kernel_size, padding=padding)(x)
            if batchnorm:
                x = BatchNormalization()(x)
            X = Activation('relu')(x)
            return X

        # 构建一个输入
        img = Input(shape=self.shape)

        # contracting path
        c1 = conv2d_block(img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout * 0.5)(p1)

        c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm, padding=padding)

        # extending path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, padding=padding)

        output = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        return Model(img, output)

    def metric_fun(self, y_true, y_pred):
        fz = tf.reduce_sum(2 * y_true * tf.cast(tf.greater(y_pred, 0.1), tf.float32)) + 1e-8
        fm = tf.reduce_sum(y_true + tf.cast(tf.greater(y_pred, 0.1), tf.float32)) + 1e-8
        return fz / fm

    def load_data(self):
        x_train = []  # 定义一个空列表，用于保存数据集
        x_label = []
        for file in glob('./train/*'):  # 获取文件夹名称
            for filename in glob(file + '/*'):  # 获取文件夹中的文件
                img = np.array(Image.open(filename), dtype='float32') / 255
                x_train.append(img[256:, 128:384])
        for file in glob('./label/*'):
            for filename in glob(file + '/*'):
                img = np.array(Image.open(filename), dtype='float32') / 255
                x_label.append(img[256:, 128:384])
        x_train = np.expand_dims(np.array(x_train), axis=3)  # 扩展维度，增加第4维
        x_label = np.expand_dims(np.array(x_label), axis=3)  # 变为网络需要的输入维度(num, 256, 256, 1)
        np.random.seed(116)  # 设置相同的随机种子，确保数据匹配
        np.random.shuffle(x_train)  # 对第一维度进行乱序
        np.random.seed(116)
        np.random.shuffle(x_label)
        # 图片有三千张左右，按9:1进行分配
        return x_train[:2700, :, :], x_label[:2700, :, :], x_train[2700:, :, :], x_label[2700:, :, :]

    def train(self, epochs=101, batch_size=32):
        os.makedirs('./weights', exist_ok=True)
        # 获得数据
        x_train, x_label, y_train, y_label = self.load_data()

        # 加载已经训练的模型
        # self.unet.load_weights(r"./best_model.h5")

        # 设置训练的checkpoint
        callbacks = [EarlyStopping(patience=100, verbose=2),
                     ReduceLROnPlateau(factor=0.5, patience=20, min_lr=0.00005, verbose=2),
                     ModelCheckpoint('./weights/best_model.h5', verbose=2, save_best_only=True)]

        # 进行训练
        results = self.unet.fit(x_train, x_label, batch_size=batch_size, epochs=epochs, verbose=2,
                                callbacks=callbacks, validation_split=0.1, shuffle=True)

        # 绘制损失曲线
        loss = results.history['loss']
        val_loss = results.history['val_loss']
        metric = results.history['metric_fun']
        val_metric = results.history['val_metric_fun']
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        x = np.linspace(0, len(loss), len(loss))  # 创建横坐标
        plt.subplot(121), plt.plot(x, loss, x, val_loss)
        plt.title("Loss curve"), plt.legend(['loss', 'val_loss'])
        plt.xlabel("Epochs"), plt.ylabel("loss")
        plt.subplot(122), plt.plot(x, metric, x, val_metric)
        plt.title("metric curve"), plt.legend(['metric', 'val_metric'])
        plt.xlabel("Epochs"), plt.ylabel("Dice")
        plt.show()  # 会弹出显示框，关闭之后继续运行
        fig.savefig('./evaluation/curve.png', bbox_inches='tight', pad_inches=0.1)  # 保存绘制曲线的图片
        plt.close()

    def test(self, batch_size=1):
        os.makedirs('./evaluation/test_result', exist_ok=True)
        self.unet.load_weights(r"weights/best_model.h5")
        # 获得数据
        x_train, x_label, y_train, y_label = self.load_data()
        test_num = y_train.shape[0]
        index, step = 0, 0
        self.unet.evaluate(y_train, y_label)
        n = 0.0
        while index < test_num:
            print('schedule: %d/%d' % (index, test_num))
            step += 1  # 记录训练批数
            mask = self.unet.predict(x_train[index:index + batch_size]) > 0.1
            mask_true = x_label[index, :, :, 0]
            if (np.sum(mask) > 0) == (np.sum(mask_true) > 0):
                n += 1
            mask = Image.fromarray(np.uint8(mask[0, :, :, 0] * 255))
            mask.save('./evaluation/test_result/' + str(step) + '.png')
            mask_true = Image.fromarray(np.uint8(mask_true * 255))
            mask_true.save('./evaluation/test_result/' + str(step) + 'true.png')
            index += batch_size
        acc = n / test_num * 100
        print('the accuracy of test data is: %.2f%%' % acc)

    def test1(self, batch_size=1):
        self.unet.load_weights(r"weights/best_model.h5")
        # 获得数据
        x_train, x_label, y_train, y_label = self.load_data()
        test_num = y_train.shape[0]
        for epoch in range(5):
            rand_index = []
            while len(rand_index) < 3:
                np.random.seed()
                temp = np.random.randint(0, test_num, 1)
                if np.sum(x_label[temp]) > 0:  # 确保产生有肿瘤的编号
                    rand_index.append(temp)
            rand_index = np.array(rand_index).squeeze()
            fig, ax = plt.subplots(3, 3, figsize=(18, 18))
            for i, index in enumerate(rand_index):
                mask = self.unet.predict(x_train[index:index + 1]) > 0.1
                ax[i][0].imshow(x_train[index].squeeze(), cmap='gray')
                ax[i][0].set_title('network input', fontsize=20)
                # 计算dice系数
                fz = 2 * np.sum(mask.squeeze() * x_label[index].squeeze())
                fm = np.sum(mask.squeeze()) + np.sum(x_label[index].squeeze())
                dice = fz / fm
                ax[i][1].imshow(mask.squeeze())
                ax[i][1].set_title('network output(%.4f)' % dice, fontsize=20)  # 设置title
                ax[i][2].imshow(x_label[index].squeeze())
                ax[i][2].set_title('mask label', fontsize=20)
            fig.savefig('./evaluation/show%d_%d_%d.png' % (rand_index[0], rand_index[1], rand_index[2]),
                        bbox_inches='tight', pad_inches=0.1)  # 保存绘制的图片
            print('finished epoch: %d' % epoch)
            plt.close()


if __name__ == '__main__':
    unet = U_Net()
    # unet.train()    # 开始训练网络
    # unet.test()     # 评价测试集并检测测试集肿瘤分割结果
    unet.test1()  # 随机显示
