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
from random import sample 

from counterexample import generate_code_dict_from_feeder, generate_images_from_codes


def grid(code, dens = 0.01):
    code_ = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    grid = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {}}
    print(dens)
    for i in range(10):
        code_[i] = np.floor(np.asarray(code[i]) / dens) - np.min(np.floor(np.asarray(code[i]) / dens), 0)
        print(code_[i])
        for pt in range(code_[i].shape[0]):
            grid[i][tuple(code_[i][pt, :])] = []
        for pt in range(code_[i].shape[0]):
            grid[i][tuple(code_[i][pt, :])].append(code[i][pt])
    return grid


def stat(grid):
    for i in range(10):
        avg_num = 0.
        tot_num = 0.
        tot_grid = 0.
        tot_dev = 0.
        std_dev = 0.
        for j in list(grid[i].keys()):
            tot_num += len(grid[i][j])
            tot_grid += 1
        avg_num = tot_num/tot_grid
        for j in list(grid[i].keys()):
            tot_dev += abs(avg_num - len(grid[i][j]))
        std_dev = tot_dev/tot_grid
        print("%i: grid %d; tot %d; avg %.5f; std %.5f" % (i, tot_grid, tot_num, avg_num, std_dev)) 
    return  tot_num, avg_num, std_dev


def preprocess(model, generator, feeder, dens):
    code_ = generate_code_dict_from_feeder(generator, feeder)
    grid_dict = grid(code_, dens)
    tot, avg, std = stat(grid_dict)
    code = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    image = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    prediction = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    loss = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    for i in range(10):
        lab = np.zeros([1, 10])
        lab[0, i] = 1

        batch_flg = 0
        for j in list(grid_dict[i].keys()):
            if len(grid_dict[i][j]) < 0.7 * (avg - std):
                continue
            if len(grid_dict[i][j]) < avg + std:
                continue
            else:
                code[i].append(np.mean(grid_dict[i][j], 0))
                #code[i].append(grid_dict[i][j][0])
                #code[i].append(np.asarray(j) + dens/2.)
                #code[i].append(sample(grid_dict[i][j], 1)[0])
                if batch_flg == generator.batch_size - 1:
                    recon = generator.decode(np.tile(lab, [generator.batch_size, 1]), code[i][-generator.batch_size:])
                    image[i] = image[i] + recon.tolist()
                    pred, los = model.predict(recon, np.tile(lab, [model.batch_size, 1]))
                    prediction[i] = prediction[i] + pred.tolist()
                    loss[i] = loss[i] + los.tolist()
                    batch_flg = 0
                else:
                    batch_flg += 1
    return code, image, prediction, loss


def hedge_attack(model, generator, feeder, save_path, dens = 0.1, T = 100, sample_num = 100, delta = 0.01, eta = 1e-3, epsilon = 1e-3):
    code, images, predictions, loss = preprocess(model, generator, feeder, dens)
    
    # Initialize sample distribution x
    xs  = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    zs  = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    recons = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    preds = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

    lab = np.eye(10)
    sample_idx  = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    # Initialize x
    for i in range(10):
        xs[i] = [1./len(code[i])] * len(code[i])
    # Start iteration
    for t in range(T):
        print("Epoch: %d" % t)
        code_t = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        # Select encoding points according to distribution x
        for i in range(10):
            sample_idx[i].append(np.random.choice(len(code[i]), 1, p = xs[i]).item())
        for i in range(10):
            zs[i].append(code[i][sample_idx[i][-1]])
            code_t[i] = zs[i][-1]

        lab_t, img_t, pred_t, loss_t, z_t = FKM_attack(model, generator, code_t, sample_num, delta, eta)
        for i in range(10):
            #zs[i][-1] = z_t[i]
            code[i][sample_idx[i][-1]] = zs[i][-1]
            recons[i].append(img_t[i])
            preds[i].append(np.argmax(pred_t[i]))
            if preds[i][-1]!= i:
                print("False prediction: %d -> %d" % (i, preds[i][-1].item(0)))
            xs[i][sample_idx[i][-1]]= xs[i][sample_idx[i][-1]]* np.exp(-epsilon * 1./xs[i][sample_idx[i][-1]] * loss_t[i]) 
            xs[i] = np.asarray(xs[i])/np.sum(xs[i])

        plot(recons, preds, zs, save_path + "_ep_%d_" % t, t)

def FKM_attack(model, generator, code, sample_num = 100, delta = 0.01, eta = 1e-3):
    z_t_ = np.reshape(list(code.values()), (10, 10))
    #plot(images, predictions, code, save_path + "_ep_0_")
    code_t = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    loss_t = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    # Initialize code_t
    for i in range(generator.label_size):
        code_t[i].append(code[i][-1])
    # Initialize search direction to zero
    g_t = np.zeros(z_t_.shape)
    # Sample next point
    for i in range(sample_num):
        # Rndom direction
        u_t_i = (np.random.random(z_t_.shape) * 2. - 1.) * delta
        # Update the encodding
        z_t_i =- z_t_ + u_t_i
        # Go through the networks
        true_label_t_i, recon_t_i, pred_t_i, loss_t_i = generate_images_from_codes(model, generator, z_t_i)
        # Calculate single search direction
        g_t += np.tile(loss_t_i, [u_t_i.shape[-1], 1]).T * u_t_i * eta / delta
    # Update the encoding with combined search directions
    z_t = z_t_ - g_t * eta 
    z_t = (np.floor(z_t_) > z_t) * np.floor(z_t) + (1. - (np.floor(z_t_)>z_t)) * z_t
    z_t = (np.ceil(z_t_) < z_t) * np.ceil(z_t) + (1. - (np.ceil(z_t_)<z_t)) * z_t
    # Go through the networks
    lab_t, img_t, pred_t, loss_t = generate_images_from_codes(model, generator, z_t)
    # Update the search direction
    return lab_t, img_t, pred_t, loss_t, z_t

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
            title = plt.title("%i: Pred %i" %  
                    (i, predictions[i][index])) 
            if int(predictions[i][index]) != i:
                plt.setp(title, color='r') 
                adv = "_cex"
            plt.colorbar()
            plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "image_" + timestamp + adv + ".png", bbox_inches="tight")
        plt.close()









if __name__ == "__main__":
    #code_path = "code_dict_latent_10_tot_100000_2020_07_12_16_39_59.pt"
    #code = pickle.load(open("./data/" + code_path, 'rb'))

    #grid_dict = grid(code, dens = 0.1)
    #tot, avg, std = stat(grid_dict)


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
    print(model_instance.batch_size)

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
    print(generator_instance.batch_size)


    mnist, size = mnist_loader()

    fig_path = "./img/adv/%s_%i/" % (args.model, args.latent_dim)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)

    fig_path = "./img/adv/%s_%i/%s/" % (args.model, args.latent_dim, timestamp)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)


    attack_opt = dict(T= args.epoch, sample_num = 100, delta = 1e-2, eta = 1e-2, epsilon = 1e-3)
    hedge_attack(model_instance, generator_instance, feeder = mnist, dens = 10., save_path = fig_path + "hedgeattack_", **attack_opt)
