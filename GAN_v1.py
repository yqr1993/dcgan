import keras.layers as kl
import keras.models as km
import keras.optimizers as ko
import keras.backend as kb
from keras.layers.advanced_activations import LeakyReLU
import os
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt


class GAN:
    def __init__(self):
        self.dis_model = None
        self.gen_model = None
        self.train_gen_model = None

        self.epoch = 300
        self.step = 100
        self.k1 = 1
        self.k2 = 1

        self.image_size = (28, 28, 1)
        self.noise_size = (100,)
        self.img_channel_scale = [127.5, 127.5, 127.5]

        self.noise_num = 32
        self.image_num = len(os.listdir("data/"))

        self.lr = 0.0002

        self.build()
        self.noises, self.y_dis, self.y_train_gen = None, None, None
        self.real, self.real_y = self.get_sample_data("data/")

    def gen_net(self):
        i_noise = kl.Input(self.noise_size, name="i_noise")
        # dense
        x = kl.Dense(128*7*7)(i_noise)
        x = kl.Reshape((7, 7, 128))(x)
        # 一套反卷积
        x = kl.Deconv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
        x = kl.BatchNormalization(momentum=0.8)(x)
        x = kl.Activation("relu")(x)
        # 一套反卷积
        x = kl.Deconv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
        x = kl.BatchNormalization(momentum=0.8)(x)
        x = kl.Activation("relu")(x)
        # 一套反卷积
        x = kl.Deconv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
        x = kl.BatchNormalization(momentum=0.8)(x)
        x = kl.Activation("relu")(x)
        # 一套反卷积
        x = kl.Deconv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),  padding="same")(x)
        x = kl.BatchNormalization(momentum=0.8)(x)
        o_gen = kl.Activation("tanh")(x)

        return km.Model(inputs=i_noise, outputs=o_gen)

    def dis_net(self):
        i_data = kl.Input(self.image_size, name="i_data")
        x = kl.Conv2D(64, (3, 3), strides=2, padding="same", name="conv1")(i_data)
        x = LeakyReLU(alpha=0.2)(x)
        x = kl.Dropout(0.25)(x)
        x = kl.Conv2D(128, (3, 3), strides=2, padding="same", name="conv2")(x)
        x = kl.ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
        x = kl.BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = kl.Dropout(0.25)(x)
        x = kl.Conv2D(256, (3, 3), strides=2, padding="same", name="conv3")(x)
        x = kl.BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = kl.Dropout(0.25)(x)
        x = kl.Conv2D(512, (3, 3), strides=1, padding="same", name="conv4")(x)
        x = kl.BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = kl.Dropout(0.25)(x)
        x = kl.Flatten()(x)
        x = kl.Dense(1, activation='sigmoid')(x)

        return km.Model(inputs=i_data, outputs=x)

    def train_gen_net(self):
        i_noise_train_gen = kl.Input(self.noise_size, name="i_noise_train_gen")
        return i_noise_train_gen

    def build(self):
        # 加载模型
        self.gen_model = self.gen_net()
        self.dis_model = self.dis_net()
        # 优化器
        optimizer = ko.Adam(self.lr, 0.5, decay=1e-8)
        # 编译模型
        self.dis_model.compile(loss="binary_crossentropy", optimizer=optimizer)
        # 创建冻结判别器
        i_noise_train_gen = self.train_gen_net()
        fakes = self.gen_model(i_noise_train_gen)
        self.dis_model.trainable = False
        o_train_gen_dis = self.dis_model(fakes)
        self.train_gen_model = km.Model(i_noise_train_gen, o_train_gen_dis)
        self.train_gen_model.compile(loss="binary_crossentropy", optimizer=optimizer)

    @staticmethod
    def set_trainability(model, trainable=False):
        for layer in model.layers:
            layer.trainable = trainable

    def img_norm(self, img):
        img = img.astype(np.float32)
        img[:, :, 0] = (img[:, :, 0] / self.img_channel_scale[0]) - 1.
        # img[:, :, 1] /= self.img_channel_scale[1]
        # img[:, :, 2] /= self.img_channel_scale[2]
        return img

    def get_noise_data(self, num_noise):
        noises = []
        y_dis = []
        y_train_gen = []
        # 加入噪声数据
        for _ in range(num_noise):
            noises.append(np.random.normal(0, 1, size=self.noise_size))
            y_dis.append([0.])
            y_train_gen.append([1.])
        self.noises = np.array(noises, dtype=np.float32)
        self.y_dis = np.array(y_dis, dtype=np.float32)
        self.y_train_gen = np.array(y_train_gen, dtype=np.float32)

    def get_sample_data(self, image_path):
        real = []
        y = []
        image_list = os.listdir(image_path)
        # 加入样本数据
        for f in image_list:
            img = imread("data/" + f, as_grey=True)
            img = np.expand_dims(img, axis=3)
            img = self.img_norm(img)
            real.append(img)
            y.append([1.])
        return np.array(real, dtype=np.float32), np.array(y, dtype=np.float32)

    def train(self):
        for e in range(self.epoch):
            self.get_noise_data(self.noise_num)
            for s in range(self.step):
                for _ in range(self.k1):
                    # 训练判别网络
                    fake_data = self.gen_model.predict(self.noises)
                    train_dis_data = np.concatenate((self.real, fake_data), axis=0)
                    y_train_dis = np.concatenate((self.real_y, self.y_dis), axis=0)
                    d_loss = self.dis_model.train_on_batch(x=train_dis_data, y=y_train_dis)
                for _ in range(self.k2):
                    # 训练生成网络
                    g_loss = self.train_gen_model.train_on_batch(x=self.noises, y=self.y_train_gen)
                print("Epoch #{} # step #{}: Generative Loss: {}, Discriminative Loss: {}".
                      format(e + 1, s + 1, g_loss, d_loss))
            # 查看生成器生成情况（图片）
            self.save_imgs(e)
        # 保存模型
        self.gen_model.save("model/gen.h5")
        print("all training finished !")
        kb.clear_session()

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.noise_size[0]))
        gen_imgs = self.gen_model.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("ret/mnist_%d.png" % epoch)
        plt.close()

    def generate(self):
        self.gen_model = km.load_model("model/gen.h5")
        ret = self.gen_model.predict(np.array([np.random.normal(0, 1, size=self.noise_size)]))[0]
        print(ret)
        imsave("ret/epoch_.jpg", ret)


if __name__ == "__main__":
    gan = GAN()
    gan.train()
