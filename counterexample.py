import logging
import argparse
import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import time
import datetime
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

from util import create_log, mnist_loader, shape_2d
from plot import get_parameter
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def create_mean_code_dict(model, generator, feeder):
    # Create coding， image and prediction dictionaries for 0～9
    code = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    images = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    predictions = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

    # Use MNIST feeder to create the initial reconstructed images
    code_ = generate_code_dict_from_feeder(generator, feeder)
    # Take the mean of each category
    z_tmp = np.vstack([np.mean(_code[_a], 0) for _a in range(generator.label_size)])
    z = np.tile(z_tmp, [int(generator.batch_size / generator.label_size), 1])
    # Generate image from each mean
    lab, img, pred, loss = generate_images_from_codes(model, generator, z)
    # Initialize the coding, image and prediction dictionaries
    for i in range(10):
        i_ = i % generator.label_size
        images[i_].append(img[i])
        predictions[i_].append(np.argmax(pred[i]))
        code[i_].append(z[i, :])
    return code, images, predictions


def noised_images_experiment(model, generator, feeder, epoch, save_path):
    code, images, predictions = create_mean_code_dict(model, generator, feeder)
    # Add noised codings, reconstruct and predict
    z = np.concatenate(list(code.values()), 0)
    for e in range(10):
        z_add_on = np.random.randn(*z.shape) * 0.01 * e
        z+= z_add_on
        lab, img, pred = generate_images_from_codes(model, generator, z)
        for i in range(generator.label_size):
            i_ = i % generator.label_size
            images[i_].append(img[i])
            predictions[i_].append(np.argmax(pred[i]))
            code[i_].append(z[i, :])
    # Draw the reconstructions and predictions
    plot(images, predictions, code, save_path)


def FKM_attack(model, generator, feeder, save_path, T = 100, sample_num = 100, delta = 0.01, eta = 1e-3):
    code, images, predictions = create_mean_code_dict(model, generator, feeder)
    z_t_ = np.concatenate(list(code.values()), 0)
    plot(images, predictions, code, save_path + "_ep_0_")
    for t in range(1, T):
        print("Epoch: %d" % t)
        code_t = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        loss_t = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        for i in range(generator.label_size):
            code_t[i].append(code[i][-1])
        g_t = np.zeros(z_t_.shape)
        for i in range(sample_num):
            u_t_i = (np.random.random(z_t_.shape) * 2. - 1.) * delta
            z_t_i =- z_t_ + u_t_i
            true_label_t_i, recon_t_i, pred_t_i, loss_t_i = generate_images_from_codes(model, generator, z_t_i)
            g_t += np.tile(loss_t_i, [u_t_i.shape[-1], 1]).T * u_t_i * eta / delta
        z_t = z_t_ - g_t * eta 
        lab_t, img_t, pred_t, loss_t = generate_images_from_codes(model, generator, z_t)
        z_t_ = np.copy(z_t)
        for i in range(generator.label_size):
            images[i].append(img_t[i])
            predictions[i].append(np.argmax(pred_t[i]))
            code[i].append(z_t[i, :])
            if i != predictions[i][-1]:
                print("False prediction: %d -> %d" % (i, predictions[i][-1].item(0)))
                print("Restart digit %d" % i)
                code_, _, _ = create_mean_code_dict(model, generator, feeder)
                z_t_[i, :] = np.copy(code_[i][0])
            

        plot(images, predictions, code, save_path + "_ep_%d_" % t, t)
        print("Update coding {}".format(g_t * eta))
        print("L2-norm {}".format(np.linalg.norm(g_t * eta, ord = 2)))

    plot(images, predictions, code, save_path)


    # Draw the reconstructions and predictions
    

def generate_code_dict_from_feeder(generator, feeder, iterations = 500, save = None):
    _code = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    _len = int(generator.batch_size / generator.label_size)
    for i in range(iterations):
        print("Generating data iteration %d" % i)
        _x, _y = feeder.test.next_batch(generator.batch_size)
        _x = shape_2d(_x, generator.batch_size)
        __code = generator.encode(_x, _y).tolist()
        for __x, __y, _c in zip(_x, _y, __code):
            _code[int(np.argmax(__y))].append(_c)
    if save is not None:
        with open(*save) as output:
             pickle.dump(_code, output)
    return _code



def generate_images_from_codes(model, generator, z):
    # convert label to one hot vector
    _len = int(generator.batch_size / generator.label_size)
    true_label_tmp = np.eye(generator.label_size)[[i for i in range(generator.label_size)]]
    true_label = np.tile(true_label_tmp, [_len, 1])
    reconstructions = generator.decode(true_label, z)
    pred, loss = model.predict(reconstructions, true_label)
    return true_label, reconstructions, pred, loss
        
def plot(images, predictions, encodings, save_path, index = None):
    if index is None:
        width = len(images[0])
    else:
        width = 1
    
    adv = ""
    plt.figure(figsize=(3 * width, 20))

    for j in range(width): 
        if width > 1 or index is None:
            index = j
        print("Processing %ith column..." % index)
        for i in range(10):
            plt.subplot(10, width, width * i + j + 1)
            plt.imshow(images[i][index].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
            title = plt.title("%i: Pred %i; Dist %.4f" %  
                    (i, predictions[i][index], 
                     np.linalg.norm(encodings[i][index] - encodings[i][0], ord = 2)))
            if int(predictions[i][index]) != i:
                plt.setp(title, color='r') 
                adv = "_cex"
            plt.colorbar()
            plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "image_" + timestamp + adv + ".png", bbox_inches="tight")
        plt.close()


def pca(path, n_comp = 5):
    code = pickle.load(open(path, 'rb'))
    print("Standardize PCA dataset")
    for i in range(10):
        code[i]= StandardScaler().fit_transform(code[i])

        pca = PCA(n_components=n_comp)
        code[i] = pca.fit_transform(code[i])
        print("Obtain pricipal components for code %d" % i)
    cols = len(np.asarray(code[0][0]).tolist())
    axis = []
    for i in range(cols - 1):
        for j in range(i+1,cols):
            axis.append([i, j])
    axis = [[0, 1], [cols - 2, cols - 1]]        
    axis = [[cols - 2, cols - 1]]

    #axis = range(cols)
    print(code[i][:10])
    visualize(code, axis)
    return code

def dump_pca_code(code, axis = None, path = None):
    vecs = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    for i in code.keys():
        for j in code[i]:
            if axis is None:
                vecs[i].append([j[-2], j[-1]])
            else:
                vecs[i].append([j[a] for a in axis])
    with open(*path) as output:
        pickle.dump(vecs, output)


    

def visualize(code, axis):
    rows = len(code.keys())
    cols = len(axis)
    dim = len(np.asarray(axis).shape)
    print(dim)
    fig = plt.figure(figsize = (12 *  cols, 6 * dim * rows))

    for i in range(rows): 
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1) 
            coords_target = []
            ax.set_xlabel('{}: Principal Component {}'.format(i, axis[j]), fontsize = 8)
            for k in range(code[i].shape[0]):
                if dim == 1:
                    coords_target.append([code[i][k][axis[j]], 0.])
                elif dim == 2:
                    coords_target.append([code[i][k][axis[j][0]], code[i][k][axis[j][1]]])
            coords_target = np.asarray(coords_target)

            ax.set_xlim(-6., + 6.)
            if dim > 1:
                ax.set_ylim(-6., + 6.)
            ax.scatter(coords_target[:, 0],coords_target[:, 1] , c = 'r', s = 5)
            #ax.legend("")
            #ax.grid()	

    fig.subplots_adjust(hspace=1., wspace = 0.5)
    plt.savefig("./img/pca/{}_{}".format(axis, timestamp))
    plt.show()

    

if __name__ == '__main__':


    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # set NumPy random seed
    np.random.seed(10)
    # set TensorFlow random seed
    tf.set_random_seed(10)
    # Parser
    parser = argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('model', action='store', nargs=None, const=None, default=None, type=str, choices=None,
                        metavar=None, help="""Name of model to use. 
- vae: variational autoencoder\n- cvae_fc3: conditional vae with 3 fully connected layer
- cvae_cnn2: conditional vae with 2 cnn\n- cvae_cnn2: conditional vae with 3 cnn """)
    parser.add_argument('-n', '--latent_dim', action='store', nargs='?', const=None, default=20, type=int,
                        choices=None, help='Latent dimension. [default: 20]', metavar=None)
    parser.add_argument('-p', '--progress', action='store', nargs='?', const=None, default=None, type=str, metavar=None,
                        choices=None, help='Use progress model (model is saved each 50 epoch). [default: None]')
    parser.add_argument('-e', '--epoch', action='store', nargs='?', const=None, default=150, type=int,
                        choices=None, help='Epoch number. [default: 150]', metavar=None)
    parser.add_argument('-s', '--std', action='store', nargs='?', const=None, default=0.01, type=float,
                        choices=None, help='Std of gaussian noise for `gen_rand` plotting. [default: 0.01]', metavar=None)
    parser.add_argument('-t', '--target', action='store', nargs='?', const=None, default=None, type=int, metavar=None,
                        choices=None,
                        help='Target digit to generate for `gen_rand` plotting. [default: None (plot all digit)]')
    parser.add_argument('-sd', '--seed', action='store', nargs='?', const=None, default=True, type=bool, metavar=None,
                        choices=None,
                        help='If use seed for `gen_rand` plotting. [default: True]')
    args = parser.parse_args()

    print("\n Adversarial attack on %s \n" % args.model)

    pr = "progress-%s-" % args.progress if args.progress else ""
    model_param = get_parameter("./parameter/%s.json" % "cnn")
    model_acc = np.load("./log/%s/%sacc.npz" % ("cnn", pr))
    model_opt = dict(network_architecture=model_param, batch_size=10)
    # opt = dict(network_architecture=param, batch_size=100)

    from model import CNN as Model
    model_path = "./log/%s/model.ckpt" % "cnn"
    model_instance = Model(load_model = model_path, **model_opt)

    generator_param = get_parameter("./parameter/%s.json" % args.model, args.latent_dim)
    generator_acc = np.load("./log/%s_%i/%sacc.npz" % (args.model, args.latent_dim, pr))
    generator_opt = dict(network_architecture=generator_param, batch_size=10, label_size = 10)

    if args.model == "cvae_cnn3_0":
        from model import CvaeCnn3_0 as GEN
    elif args.model == "cvae_cnn3":
        from model import CvaeCnn3 as GEN
    elif args.model == "cvae_cnn2":
        from model import CvaeCnn2 as GEN
    elif args.model == "cvae_fc3":
        from model import CvaeFc3 as GEN
    elif args.model == "vae":
        from model import VariationalAutoencoder as GEN
    generator_path = "./log/%s_%i/%smodel.ckpt" % (args.model, args.latent_dim, pr)
    generator_instance = GEN(load_model = generator_path, **generator_opt)


    mnist, size = mnist_loader()

    code_path = "./data/code_dict_latent_%i_tot_%i_%s" % (args.latent_dim, args.epoch * 10, timestamp)
    generate_code_dict_from_feeder(generator = generator_instance,  
                                    feeder = mnist, iterations = args.epoch, 
                                    save = (code_path, 'wb'))
    code = pca(code_path, 10)
    dump_pca_code(code, (8, 9), (code_path + "_pca_{}".format((8, 9)), "wb"))
    exit(0)

    fig_path = "./img/adv/%s_%i/" % (args.model, args.latent_dim)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)

    fig_path = "./img/adv/%s_%i/%s/" % (args.model, args.latent_dim, timestamp)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)

    #noised_images_experiment(model=model_instance, generator=generator_instance, feeder = mnist, epoch=args.epoch, save_path = fig_path + "noised_")
    
    #attack_opt = dict(T = args.epoch, sample_num = 100, delta = 1e-2, eta = 1e-2)
    #FKM_attack(model=model_instance, generator=generator_instance, feeder = mnist, save_path = fig_path + "attack_", **attack_opt)






