import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from utils.vgg_model import VggEncoder
from utils.lightweight_model import VggSDecoder, VggSEncoder
from utils.photo_gif import GIFSmoothing
from PIL import Image
from tqdm import tqdm
import os, cv2

class VggEncDec(tf.keras.Model):
    def __init__(self):
        super(VggEncDec, self).__init__()
        self.sencoder = VggSEncoder()
        self.sdecoder = VggSDecoder()
        self.sencoder.load_weights('ckpts/sencoder')
        self.sdecoder.load_weights('ckpts/sdecoder')

    def call(self, layer, input_img):
        None
        
def load_img(file):
    img = np.asarray(Image.open(file), dtype=np.float32)
    img = np.expand_dims(cv2.resize(img, (img.shape[1] // 8 * 8, img.shape[0] // 8 * 8)), axis=0) / 255
    return img

def inv_sqrt_cov(cov, inverse=False):
    s, u, _ = tf.linalg.svd(cov + tf.eye(cov.shape[-1])) 
    n_s = tf.reduce_sum(tf.cast(tf.greater(s, 1e-5), tf.int32))
    s = tf.sqrt(s[:,:n_s])
    if inverse:
        s = 1 / s
    d = tf.linalg.diag(s)
    u = u[:,:,:n_s]
    return tf.matmul(u, tf.matmul(d, u, adjoint_b=True))

# transform: feature transformation
def stylize_core(c_feat, s_feat, transform='zca'):
    n_batch, cont_h, cont_w, n_channel = c_feat.shape
    _c_feat = tf.reshape(tf.transpose(c_feat, [0, 3, 1, 2]), [n_batch, n_channel, -1])
    if transform == 'zca':
        c_feat = stylize_zca(_c_feat, s_feat) 
    elif transform == 'ot':
        c_feat = stylize_ot(_c_feat, s_feat) 
    elif transform == 'adain':
        c_feat = stylize_adain(_c_feat, s_feat) 
    
    c_feat = tf.transpose(tf.reshape(c_feat, [n_batch, n_channel, cont_h, cont_w]), [0, 2, 3, 1])
    return c_feat

def stylize_adain(c_feat, s_feat):
    m_c = tf.reduce_mean(c_feat, axis=-1, keepdims=True)
    m_s = tf.reduce_mean(s_feat, axis=-1, keepdims=True) 
    c_feat = c_feat - m_c
    s_feat = s_feat - m_s 
    s_c = tf.sqrt(tf.reduce_mean(c_feat * c_feat, axis=-1, keepdims=True) + 1e-8)
    s_s = tf.sqrt(tf.reduce_mean(s_feat * s_feat, axis=-1, keepdims=True) + 1e-8) 
    white_c_feat = c_feat / s_c
    feat = white_c_feat * s_s + m_s     
    return feat

def stylize_zca(c_feat, s_feat): 
    m_c = tf.reduce_mean(c_feat, axis=-1, keepdims=True)
    m_s = tf.reduce_mean(s_feat, axis=-1, keepdims=True) 
    c_feat = c_feat - m_c
    s_feat = s_feat - m_s
    c_cov = tf.matmul(c_feat, c_feat, transpose_b=True) / c_feat.shape[-1]
    s_cov = tf.matmul(s_feat, s_feat, transpose_b=True) / s_feat.shape[-1] 
    inv_sqrt_c_cov = inv_sqrt_cov(c_cov, True)
    opt = tf.matmul(inv_sqrt_cov(s_cov), inv_sqrt_c_cov) 
    feat = tf.matmul(opt, c_feat) + m_s 
    return feat

def stylize_ot(c_feat, s_feat):
    m_c = tf.reduce_mean(c_feat, axis=-1, keepdims=True)
    m_s = tf.reduce_mean(s_feat, axis=-1, keepdims=True) 
    c_feat = c_feat - m_c
    s_feat = s_feat - m_s 
    c_cov = tf.matmul(c_feat, c_feat, transpose_b=True) / c_feat.shape[-1]
    s_cov = tf.matmul(s_feat, s_feat, transpose_b=True) / s_feat.shape[-1] 
    sqrt_c_cov = inv_sqrt_cov(c_cov)
    inv_sqrt_c_cov = inv_sqrt_cov(c_cov, True)
    opt = inv_sqrt_cov(tf.matmul(sqrt_c_cov, tf.matmul(s_cov, sqrt_c_cov))) 
    opt = tf.matmul(inv_sqrt_c_cov, tf.matmul(opt, inv_sqrt_c_cov))
    feat = tf.matmul(opt, c_feat) + m_s 
    return feat


def run(progress_callback = None, seed=0):

    content = './examples/content'
    style = './examples/style'
    output = './outputs'
    output_list = './outputs_list'

    if not os.path.exists(os.path.join(output)):
        os.makedirs(os.path.join(output))
    
    if not os.path.exists(os.path.join(output_list)):
        os.makedirs(os.path.join(output_list))

    enc_dec = VggEncDec()
    p_pro = GIFSmoothing(r=30, eps= (0.02 * 255) ** 2)

    cont_seed = f'{seed}_target.jpg'
    style_seed = f'{seed}_concat_image.jpg'
    cont_img = load_img(os.path.join(content, cont_seed))
    style_img = load_img(os.path.join(style, style_seed))
    
    transform = 'zca'

    x1 = enc_dec.sencoder(0, style_img)
    x2 = enc_dec.sencoder(1, x1[0])
    x3 = enc_dec.sencoder(2, x2[0])
    x4 = enc_dec.sencoder(3, x3[0])

    y1 = enc_dec.sencoder(0, cont_img)
    y2 = enc_dec.sencoder(1, y1[0])
    y3 = enc_dec.sencoder(2, y2[0])
    y4 = enc_dec.sencoder(3, y3[0])
    progress_callback(0.2)
    
    sfeat = tf.reshape(tf.transpose(x4[0], [0, 3, 1, 2]), [x4[0].shape[0], x4[0].shape[-1], -1])
    x = stylize_core(y4[0], sfeat, transform=transform)
    x = enc_dec.sdecoder(3, x, skip=y4[1])
    progress_callback(0.4)

    sfeat = tf.reshape(tf.transpose(x3[0], [0, 3, 1, 2]), [x3[0].shape[0], x3[0].shape[-1], -1])
    x = stylize_core(x, sfeat, transform=transform)
    x = enc_dec.sdecoder(2, x, skip=y3[1])
    progress_callback(0.6)

    sfeat = tf.reshape(tf.transpose(x2[0], [0, 3, 1, 2]), [x2[0].shape[0], x2[0].shape[-1], -1])
    x = stylize_core(x, sfeat, transform=transform)
    x = enc_dec.sdecoder(1, x, skip=y2[1])
    progress_callback(0.8)

    sfeat = tf.reshape(tf.transpose(x1[0], [0, 3, 1, 2]), [x1[0].shape[0], x1[0].shape[-1], -1])
    x = stylize_core(x, sfeat, transform=transform)
    x = tf.clip_by_value(enc_dec.sdecoder(0, x, skip=y1[1]), 0, 1)
    progress_callback(0.99)

    p_pro.process(x[0], os.path.join(content, cont_seed)).save(os.path.join(output, f'{seed}_result.jpg'))
    
    # image_A = Image.open(os.path.join(content, cont_seed))
    # image_B = Image.open(os.path.join(output, f'{seed}_result.jpg'))
    # image_np_A = np.array(image_A) 
    # image_np_B = np.array(image_B)
    # image_np_B = cv2.resize(image_np_B, (image_np_A.shape[1], image_np_A.shape[0]))

    # for i in range(10):
    #     t = 10
    #     x = i/t
    #     new_image_np = image_np_A * (1-x) + image_np_B * (x)
    #     new_image_np = np.clip(new_image_np, 0, 255).astype(np.uint8)
    #     new_image = Image.fromarray(new_image_np)
    #     output_image_path = os.path.join(output_list, f'{seed}_{i}_result.png')
    #     new_image.save(output_image_path)

if __name__ =='__main__':
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"  
    run()