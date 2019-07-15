# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from imageio import imread, imsave
import os, glob, cv2

no_makeup = os.path.join('faces', 'no_makeup', 'xfsy_0055.png')
makeup_a = os.path.join('faces', 'makeup', 'XMY-074.png')
makeup_b = os.path.join('faces', 'makeup', 'vFG112.png')
img_size = 256

class DMT(object):
    def __init__(self):
        self.pb = 'dmt.pb'
        self.style_dim = 8

    def preprocess(self, img):
        return (img / 255. - 0.5) * 2

    def deprocess(self, img):
        return (img + 1) / 2

    def load_image(self, path):
        img = cv2.resize(imread(path), (img_size, img_size))
        img_ = np.expand_dims(self.preprocess(img), 0)
        return img / 255., img_

    def load_model(self):
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()

            with open(self.pb, 'rb') as fr:
                output_graph_def.ParseFromString(fr.read())
                tf.import_graph_def(output_graph_def, name='')

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            graph = tf.get_default_graph()
            self.X = graph.get_tensor_by_name('X:0')
            self.Y = graph.get_tensor_by_name('Y:0')
            self.S = graph.get_tensor_by_name('S:0')
            self.X_content = graph.get_tensor_by_name('content_encoder/content_code:0')
            self.X_style = graph.get_tensor_by_name('style_encoder/style_code:0')
            self.Xs = graph.get_tensor_by_name('decoder_1/g:0')
            self.Xf = graph.get_tensor_by_name('decoder_2/g:0')

    # 成对妆容迁移
    def pairwise(self, A, B):
        print('成对妆容迁移...')
        A_img, A_img_ = self.load_image(A)
        B_img, B_img_ = self.load_image(B)
        Xs_ = self.sess.run(self.Xs, feed_dict={self.X: A_img_, self.Y: B_img_})

        result = np.ones((img_size, 3 * img_size, 3))
        result[:, :img_size] = A_img
        result[:, img_size: 2 * img_size] = B_img
        result[:, 2 * img_size:] = self.deprocess(Xs_)[0]
        imsave(os.path.join('output', 'pairwise.jpg'), result)

    # 插值妆容迁移
    def interpolated(self, A, B, n=3):
        print('插值妆容迁移...')
        A_img, A_img_ = self.load_image(A)
        B_img, B_img_ = self.load_image(B)
        A_style = self.sess.run(self.X_style, feed_dict={self.X: A_img_})
        B_style = self.sess.run(self.X_style, feed_dict={self.X: B_img_})

        result = np.ones((img_size, (n + 3) * img_size, 3))
        result[:, :img_size] = A_img
        result[:, (n + 2) * img_size:] = B_img

        for i in range(n + 1):
            Xf_ = self.sess.run(self.Xf, feed_dict={self.X: A_img_, self.S: (n - i) / n * A_style + i / n * B_style})
            result[:, (i + 1) * img_size: (i + 2) * img_size] = self.deprocess(Xf_)[0]
        imsave(os.path.join('output', 'interpolated.jpg'), result)

    # 混合妆容迁移
    def hybrid(self, A, B1, B2, n=3):
        print('混合妆容迁移...')
        A_img, A_img_ = self.load_image(A)
        B1_img, B1_img_ = self.load_image(B1)
        B2_img, B2_img_ = self.load_image(B2)
        B1_style = self.sess.run(self.X_style, feed_dict={self.X: B1_img_})
        B2_style = self.sess.run(self.X_style, feed_dict={self.X: B2_img_})

        result = np.ones((img_size, (n + 3) * img_size, 3))
        result[:, :img_size] = B1_img
        result[:, (n + 2) * img_size:] = B2_img

        for i in range(n + 1):
            Xf_ = self.sess.run(self.Xf, feed_dict={self.X: A_img_, self.S: (n - i) / n * B1_style + i / n * B2_style})
            result[:, (i + 1) * img_size: (i + 2) * img_size] = self.deprocess(Xf_)[0]
        imsave(os.path.join('output', 'hybrid.jpg'), result)

    # 多模态妆容迁移
    def multimodal(self, A, n=3):
        print('多模态妆容迁移...')
        A_img, A_img_ = self.load_image(A)       
        limits = [
            [0.21629652, -0.43972224],
            [0.15712686, -0.44275892],
            [0.36736163, -0.2079917],
            [0.16977102, -0.49441707],
            [0.2893533, -0.25862852],
            [0.69064325, -0.11329838],
            [0.31735066, -0.48868555],
            [0.50784767, -0.08443227]
        ]
        result = np.ones((n * img_size, n * img_size, 3))

        for i in range(n):
            for j in range(n):
                S_ = np.ones((1, 1, 1, self.style_dim))
                for k in range(self.style_dim):
                    S_[:, :, :, k] = np.random.uniform(low=limits[k][1], high=limits[k][0])
                Xf_ = self.sess.run(self.Xf, feed_dict={self.X: A_img_, self.S: S_})
                result[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size] = self.deprocess(Xf_)[0]
        imsave(os.path.join('output', 'multimodal.jpg'), result)

if __name__ == '__main__':
    model = DMT()
    model.load_model()
    model.pairwise(no_makeup, makeup_a)
    model.interpolated(no_makeup, makeup_a)
    model.hybrid(no_makeup, makeup_a, makeup_b)
    model.multimodal(no_makeup)