'''
Sketch2Normal implementation using Tensorflow for paper
Interactive Sketch-Based Normal Map Generation with Deep Neural Networks
Author Wanchao Su
This code is based on the implementation of pix2pix from  https://github.com/yenchenlin/pix2pix-tensorflow
'''
import os
import time
import random
import numpy as np
from glob import glob
import scipy.misc
from ops import *
import vgg16
import vgg19
import skimage.transform
class normalnet(object):
    def __init__(self, sess, image_size=256,
                 batch_size=1, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=100, n_critic=5, clamp=0.01,
                 input_c_dim=3, output_c_dim=3, dataset_name='primitive', coefficient=100,phase='train'):

        self.sess = sess
        self.clamp = clamp#wgan
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size
        self.phase=phase

        self.gf_dim = gf_dim#64
        self.df_dim = df_dim#64
        self.kl_weight=5e5

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda #100
        self.coefficient = coefficient #100
        self.n_critic = n_critic #5
        #判别器

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        #生成器
        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        #vae
        self.g_bn_vae2 = batch_norm(name='g_bn_vae2')
        self.g_bn_vae3 = batch_norm(name='g_bn_vae3')
        self.g_bn_vae4 = batch_norm(name='g_bn_vae4')
        self.g_bn_vae5 = batch_norm(name='g_bn_vae5')
        self.g_bn_vae6 = batch_norm(name='g_bn_vae6')
        self.g_bn_vae7 = batch_norm(name='g_bn_vae7')
        self.g_bn_vae8 = batch_norm(name='g_bn_vae8')

        self.dataset_name = dataset_name
        self.build_model()
    def resize(self,img):
        # img=img.eval()
        # img = img / 255.0  # 归一化
        img=tf.image.resize_images(img,[224,224])
        # assert (0 <= img).all() and (img <= 1.0).all()  # 判断是否归一化成功
        # print "Original Image Shape: ", img.shape
        # we crop image from center
        # short_edge = min(img.shape[:2])
        # print("!!!!!!!!!", short_edge)  # 360
        # print(img.shape[0])  # 360
        # print(img.shape[1])  # 480

        # yy = int((img.shape[0] - short_edge) / 2)
        # print(yy, "YY")
        # xx = int((img.shape[1] - short_edge) / 2)
        # print(xx, "XX")
        # crop_img = img[yy: yy + short_edge, xx: xx + short_edge]  # 相当于从图片的最中心取出[short_edge,short_edg]大小图片
        # resize to 224, 224
        # print(crop_img.shape[0], "第一个")
        # print(crop_img.shape[1], "第二个")

        # resized_img = skimage.transform.resize(img, (224, 224))  # vgg的输入大小是224*224的
        # image = resized_img.reshape((1, 224, 224, 3))
        # image = tf.convert_to_tensor(image)
        return img


    def build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')
        #self.real_A 是对应的草图
        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        #self.real_B 是对应的relief的高度
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        self.fiture_all_loss=0
        self.real_AB = tf.concat([self.real_A, self.real_B], 3)  # 1*256*256*6
        # self.real_AB_size=tf.image.resize_images(self.real_AB,[64,64])
        # print(self.real_AB_size.shape)#(1, 64, 64, 6)
        # self.z,self.kl_loss=self.vae(self.real_AB)


        self.fake_B = self.generator(self.real_A)

        self.fake_B=tf.concat([self.fake_B,self.fake_B,self.fake_B],3)#用草图生成出来的relief

        self.resize_real_B=self.resize(self.real_B)
        self.resize_fake_B=self.resize(self.fake_B)
        # self.resize_real_A = self.resize(self.real_A)
        if self.phase=='train':
            vgg = vgg19.Vgg19() #获得了数据
            self.real_f1,self.real_f2,self.real_f3,self.real_f4,self.real_f5=vgg.build(self.resize_real_B)
            self.fake_f1,self.fake_f2,self.fake_f3,self.fake_f4,self.fake_f5 =vgg.build(self.resize_fake_B)
            self.f1_loss = tf.reduce_mean(tf.abs(self.real_f1 - self.fake_f1))
            self.f2_loss = tf.reduce_mean(tf.abs(self.real_f2 - self.fake_f2))
            self.f3_loss = tf.reduce_mean(tf.abs(self.real_f3 - self.fake_f3))
            self.f4_loss = tf.reduce_mean(tf.abs(self.real_f4 - self.fake_f4))
            self.f5_loss = tf.reduce_mean(tf.abs(self.real_f5 - self.fake_f5))
            self.fiture_all_loss = 100 / (64. * 64.) * self.f1_loss + 100 / (128. * 128) * self.f2_loss + 100 / (
                        256. * 256) * self.f3_loss + 100 / (512. * 512.) * self.f4_loss + 100 / (
                                               512. * 512.) * self.f5_loss
            self.fiture_all_loss=self.fiture_all_loss*1000




        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)  # 1*224*224*67

        # self.real_AB = tf.concat([self.resize_real_A, self.real_f1], 3)#1*224*224*67
        # print("!!!!!!2@@@@@@@@@@@",self.real_AB.shape)
        # self.fake_AB = tf.concat([self.resize_real_A, self.fake_f1], 3)
        # print("!!!!!!2!!!!!!!!!!!", self.fake_AB.shape)

        # self.D,self.D_logits = self.discriminator(self.real_AB, reuse=False)
        # self.D_,self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)
        self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

        # self.fake_B_sample = self.generator(self.real_A, isSampling=True)
        # self.z, self.kl = self.vae(self.real_AB_size,isSampling=True)
        self.fake_B_sample = self.generator(self.real_A, isSampling=True)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)
        self.real_B_sum = tf.summary.image("real_B", self.real_B)


        self.d_loss_real = tf.reduce_mean(self.D_logits)
        self.d_loss_fake = tf.reduce_mean(self.D_logits_)
        # self.d_loss_real = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        # self.d_loss_fake = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))

        epsilon = tf.random_uniform([], 0.0, 1.0)
        self.x_hat = epsilon * self.fake_B + (1 - epsilon) * self.real_B
        self.x_hat_=tf.concat([self.real_A, self.x_hat], 3)
        d_hat__ = self.discriminator(self.x_hat_,reuse=True)
        self.ddx = tf.gradients(d_hat__, self.x_hat)[0]
        self.ddx = tf.sqrt(tf.reduce_sum(tf.square(self.ddx), axis=1))
        self.ddx = tf.reduce_mean(tf.square(self.ddx - 1.0) * 10.0)

        self.pixel_wised_loss =self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))


        self.d_loss = self.d_loss_fake - self.d_loss_real +self.ddx
        # self.g_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
        #               + self.L1_lambda * self.pixel_wised_loss+self.fiture_all_loss
        # self.kl_loss=self.kl_weight*self.kl_loss

        self.g_loss = -self.d_loss_fake \
                      +  self.pixel_wised_loss + self.fiture_all_loss#+self.kl_loss


        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.fiture_loss_sum = tf.summary.scalar("fiture_loss", self.fiture_all_loss)


        self.pixel_wised_loss_sum = tf.summary.scalar("pixeled_loss", self.pixel_wised_loss)
        # self.kl_loss_sum = tf.summary.scalar("kl_loss", self.kl_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def load_random_samples(self):
        data = np.random.choice(glob('{}/val/*.png'.format(self.dataset_name)), self.batch_size)#随机选取一个图片路径
        sample = [self.load_data(sample_file) for sample_file in data]
        sample_images = np.array(sample).astype(np.float32)
        return sample_images


    def sample_model(self, sample_dir, epoch, idx):
        sample_images = self.load_random_samples()#
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images}
        )
        self.save_images(samples, [self.batch_size, 1], './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            return  h4

            # return  tf.nn.sigmoid(h4),h4
    def vae(self,image,isSampling=False):
        with tf.variable_scope("generator_vae") as scope:

            if isSampling:
                scope.reuse_variables()



            # image is (256 x 256 x 6)
            image=(image+1.0)*127.5
            print("111",image.shape)
            e1 = batch_norm2(conv2d(image, 2, name='g_vae1_conv'),'g_vae1_conv')
            e2 = batch_norm2(conv2d(lrelu(e1), 4, name='g_vae2_conv'),'g_vae2_conv')
            e3 = batch_norm2(conv2d(lrelu(e2), 8, name='g_vae3_conv'),'g_vae3_conv')
            e4 = batch_norm2(conv2d(lrelu(e3), 16, name='g_vae4_conv'),'g_vae4_conv')
            e5 = batch_norm2(conv2d(lrelu(e4), 32, name='g_vae5_conv'),'g_vae5_conv')
            e6 = batch_norm2(conv2d(lrelu(e5), self.gf_dim * 8, name='g_vae6_conv'),'g_vae6_conv')
            e7 = batch_norm2(conv2d(lrelu(e6), self.gf_dim * 8, name='g_vae7_conv'),'g_vae7_conv')

            e8 = batch_norm2(conv2d(lrelu(e7), self.gf_dim * 8*2, name='g_vae8_conv'),'g_vae8_conv')
            e8 = tf.nn.relu(e8)


            mu=e8[:,:,:,:self.gf_dim * 8] #均值
            # print(mu.shape)
            logvar=e8[:,:,:,self.gf_dim * 8:] #方差的对数 logσ2

            epsilon = tf.random_normal(tf.shape(logvar), name='epsilon')
            var = tf.exp(0.5 * logvar) #σ 标准差

            z = mu + var * epsilon  # 得到z

            # tf.reduce_sum(0.5 * (tf.square(mu) + tf.square(var) -
            #                      2.0 * tf.log(var + 1e-8) - 1.0))

            obj_kl=tf.reduce_sum(0.5 * (tf.square(mu) + tf.square(var) -
                                 2.0 * tf.log(var + 1e-8) - 1.0),axis = [1,2,3])


            # obj_kl = tf.reduce_sum(mu * mu / 2.0 - logvar + tf.exp(logvar) / 2.0 - 0.5, axis = [1,2,3])
            # obj_kl=tf.reshape(image,[-1,256*256*6])
            kl_loss = tf.reduce_mean(obj_kl)  # kl loss
            return z,kl_loss




    def generator(self, image,isSampling=False):
        with tf.variable_scope("generator") as scope:

            if isSampling:
                scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
            # image = tf.concat([image, image], 3)
            # self.z=z

            # image is (256 x 256 x 3)  gf_dim=64=output_dim 编码器
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))

            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            e8=tf.nn.relu(e8)

            # print(hiddel.type)

            # e8=tf.concat([e8,z],3)
            #e8 is (1 x 1 x self.gf_dim * 8)

            #解码器
            self.d1, self.d1_w, self.d1_b = deconv2d(e8,[self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            #先进行convd操作
            # self.d1 = conv2d(tf.nn.relu(e8),output_dim=self.gf_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1,name='g_d1')

            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),[self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            # self.d2 = conv2d(tf.nn.relu(d1), output_dim=self.gf_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d2')
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),[self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            # self.d3 = conv2d(tf.nn.relu(d2), output_dim=self.gf_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d3')
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),[self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            # self.d4 = conv2d(tf.nn.relu(e8), output_dim=self.gf_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d')
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, 1], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)


    def save(self, checkpoint_dir, step):
        model_name = "sketch2normal.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def train(self, args):

        self.d_optim = tf.train.AdamOptimizer(learning_rate=5e-6, beta1=0.5,beta2=0.9).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5,beta2=0.9).minimize(self.g_loss, var_list=self.g_vars)



        # self.d_optim = tf.train.RMSPropOptimizer(learning_rate=args.lr).minimize(self.d_loss, var_list=self.d_vars)
        # self.g_optim = tf.train.RMSPropOptimizer(learning_rate=args.lr).minimize(self.g_loss, var_list=self.g_vars)
        # self.clip_d_vars_ops = [val.assign(tf.clip_by_value(val, -self.clamp, self.clamp)) for val in self.d_vars]
        tf.global_variables_initializer().run()
        # self.clip_d_vars_ops = [val.assign(tf.clip_by_value(val, -self.clamp, self.clamp)) for val in self.d_vars]
        # tf.global_variables_initializer().run()

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_summary = tf.summary.merge([self.fake_B_sum, self.real_B_sum,self.d_loss_fake_sum, self.g_loss_sum,self.fiture_loss_sum])
        self.d_summary = tf.summary.merge([self.d_loss_real_sum, self.d_loss_sum])
        #self.visual_loss_summary = tf.summary.merge([self.pixel_wised_loss_sum, self.masked_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(args.epoch):#
            data = glob('{}/train/*.png'.format(self.dataset_name)) #获得名字保存在data中
            np.random.shuffle(data)#打乱顺序 名字

            batch_idxs = min(len(data), 1e8) // (self.batch_size*self.n_critic)#计算出步数
            print('[*] run optimizor...')

            for idx in range(0, batch_idxs):#
                errD=.0
                batch_list = [self.load_training_imgs(data, idx+i) for i in range(self.n_critic)] #一次从data中从idx位置开始取batch个图片一共取五次

                for j in range(self.n_critic):
                    batch_images = batch_list[j] #

                    _, errD, errd_real, errd_fake,  summary_str = self.sess.run([self.d_optim, self.d_loss,
                                                                                self.d_loss_real, self.d_loss_fake,
                                                                                self.d_summary],
                                                                                           feed_dict={self.real_data: batch_images})
                    # self.sess.run(self.clip_d_vars_ops)#wgan 需要进行截断
                    self.writer.add_summary(summary_str, counter)
                    #self.writer.add_summary(errVis_sum, counter)

                # Update G network
                _, errG, summary_str,vgg_loss ,pix_loss,kl= self.sess.run([self.g_optim, self.g_loss, self.g_summary,self.fiture_all_loss,self.pixel_wised_loss,self.kl_loss],
                                               feed_dict={self.real_data: batch_list[np.random.randint(0, self.n_critic, size=1)[0]]})#从batch_list中随机取一张图片
                self.writer.add_summary(summary_str, counter)

                _, errG, summary_str ,vgg_loss,pix_loss,kl= self.sess.run([self.g_optim, self.g_loss, self.g_summary,self.fiture_all_loss,self.pixel_wised_loss,self.kl_loss],
                                                     feed_dict={self.real_data: batch_list[
                                                         np.random.randint(0, self.n_critic, size=1)[0]]})
                self.writer.add_summary(summary_str, counter)

                # print("!!!!!!!!!",self.sess.run(self.fiture_all_loss))

                current = time.time()
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, vgg_loss: %.8f, pix_loss :%.8f, kl_loss :%.8f" \
                      % (epoch, idx, batch_idxs, current - start_time, errD, errG,vgg_loss,pix_loss,kl))
                start_time = current

                if np.mod(counter, 100) == 1:
                    self.sample_model(args.sample_dir, epoch, idx) #

                if np.mod(counter, 1000) == 2:
                    self.save(args.checkpoint_dir, counter)
                counter += 1

    def test(self, args):

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        test_files = glob('{}/test/*.png'.format(self.dataset_name))
        print(test_files)

        n = [int(i) for i in map(lambda x: x.split('\\')[-1].split('.png')[0], test_files)]
        test_files = [x for (y, x) in sorted(zip(n, test_files))]

        # load testing input
        print("Loading testing images ...")
        images = [self.load_data(file, is_test=True) for file in test_files]
        # print(images[-1])

        test_images = np.array(images).astype(np.float32)
        # print(test_images.shape)
        test_images = [test_images[i:i+self.batch_size] for i in range(0, len(test_images), self.batch_size)]
        # print(test_images.shape)
        test_images = np.array(test_images)
        # print(test_images.shape[-1])

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for i, test_image in enumerate(test_images):
            idx = i+1
            print("test image " + str(idx))
            if test_image.shape[-1]==3:
                tf.concat([test_image,test_image],-1)

            results = self.sess.run(self.fake_B_sample, feed_dict={self.real_data: test_image})
            self.save_images(results, [self.batch_size, 1], './{}/test_{:04d}.png'.format(args.test_dir, idx))

    def load_training_imgs(self, data, idx):
        batch_files = data[idx * self.batch_size:(idx + 1) * self.batch_size] #这里取第idx张图片放入batch_files
        batch = [self.load_data(batch_file) for batch_file in batch_files] #返回图片
        # batch2 = [self.load_data(batch_file) for batch_file in batch_files]  # 返回图片

        batch_images = np.reshape(np.array(batch).astype(np.float32),
                                  (self.batch_size, self.image_size, self.image_size, -1))

        return batch_images

    def save_images(self, images, size, image_path):
        images = (images + 1) / 2.0 #图片归一化
        h, w = images.shape[1], images.shape[2]
        # 产生一个大画布，用来保存生成的 batch_size 个图像
        img = np.zeros((h * size[0], w * size[1], 3))
        # 循环使得画布特定地方值为某一幅图像的值
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        scipy.misc.imsave(image_path, img)

    def load_data(self, image_path, is_test=False,is_unet=True):
        input_img = scipy.misc.imread(image_path).astype(np.float)#读取图片


        # index = image_path.find(self.dataset_name + '/')
        #print(index)
        # insert_point = index + len(self.dataset_name) + 1
        # mask_path = image_path[:insert_point] + 'mask/' + image_path[insert_point:]#获得mask的路径
        # mask = scipy.misc.imread(mask_path, mode='L').astype(np.float)
        img_A = input_img[:, 0:256, :] #草图
        img_B = input_img[:, 256:512, :] #relief

        if not is_test: #
            num = random.randint(256, 286) #在256-286中间随机产生一个数
            img_A = scipy.misc.imresize(img_A, (num, num))#
            img_B = scipy.misc.imresize(img_B, (num, num))
            #mask = scipy.misc.imresize(mask, (num, num))
            num_1 = random.randint(0, num - 256)
            img_A = img_A[num_1:num_1 + 256, num_1:num_1 + 256, :] #再重新选取256*256大小
            img_B = img_B[num_1:num_1 + 256, num_1:num_1 + 256, :]
            #mask = mask[num_1:num_1 + 256, num_1:num_1 + 256]

        #mask = np.reshape(mask, (256, 256, 1)) / 255.0
        if is_unet:
            img_A = img_A / 127.5 - 1.
            img_B = img_B / 127.5 - 1.
            img_AB = np.concatenate((img_A, img_B), axis=2)
        else:
            img_AB = np.concatenate((img_A, img_B), axis=2)



        # print(img_AB.shape)
        # if img_AB.shape[-1]==8:
        #      img_AB=tf.concat([img_AB,img_AB,img_AB],-1)


        return img_AB
