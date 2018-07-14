import numpy as np
import tensorflow as tf

import vgg16
import utils
#加载图片
#img1 = utils.load_image("./test_data/timg.jpeg")#路径
img2 = utils.load_image("./test_data/cat.jpg")

#batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

#batch = np.concatenate((batch1, batch2), 0)

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/gpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [1, 224, 224, 3])#占位符
        feed_dict = {images: batch2}

        vgg = vgg16.Vgg16()#
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        #print(prob)
        utils.print_prob(prob[0], './synset.txt')
        # utils.print_prob(prob[1], './synset.txt')
