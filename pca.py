import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
import time
import datetime
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')


def pca(path, n_comp = 5):
    code = pickle.load(open(path, 'rb'))
    mean = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    inf = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    sup = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    comp = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    print("Standardize PCA dataset")
    scalar= StandardScaler()
    for i in range(10):
        inf[i] = np.min(code[i], 0)
        sup[i] = np.max(code[i], 0)

        scalar.fit(code[i])
        mean[i] = scalar.mean_
        code[i] = (np.asarray(code[i]) - mean[i])/(sup[i] - inf[i])

        pca = PCA(n_components=n_comp)
        code[i] = pca.fit_transform(code[i])
        comp[i] = pca.components_
        print("Obtain pricipal components for code %d" % i)
        
    return code, mean, sup, inf, comp

def pca_reverse(code, mean, sup, inf, comp):
    for i in range(10):
        code[i] = np.dot(np.asarray(code[i]), comp[i])*(sup[i] - inf[i]) + mean[i] 


def visualize(code, axis = None, path = None):
    rows = len(code.keys())
    if axis is not None:
        cols = len(axis)
    else:
        cols = len(code[0])
        axis = range(cols) 
    dim = len(np.asarray(axis).shape)
    fig = plt.figure(figsize = (12 *  cols, 6 * dim * rows))

    for i in range(rows): 
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1) 
            coords_target = []
            ax.set_xlabel('{}: Principal Component {}'.format(i, axis[j]), fontsize = 8)
            for k in range(np.asarray(code[i]).shape[0]):
                if dim == 1:
                    coords_target.append([code[i][k][axis[j]], 0.])
                elif dim == 2:
                    coords_target.append([code[i][k][axis[j][0]], code[i][k][axis[j][1]]])
            coords_target = np.asarray(coords_target)

            ax.set_xlim(-1., + 1.)
            if dim > 1:
                ax.set_ylim(-1., + 1.)
            ax.scatter(coords_target[:, 0],coords_target[:, 1] , c = 'r', s = 5)

    fig.subplots_adjust(hspace=1., wspace = 0.5)
    if path is not None:
        path = "./img/pca/{}_{}".format(axis, timestamp)
    plt.savefig(path)
    plt.show()

if __name__ == '__main__':
    code_path = "code_dict_latent_10_tot_100000_2020_07_12_16_40_32"
    code, mean, sup, inf, comp = pca("./data/" + code_path, 10)

    save_dict= dict(code = code, mean = mean, sup = sup, inf = inf, comp = comp)
    path = ("./data/" + code_path + "_pca", "wb")
    with open(*path) as output:
        pickle.dump(save_dict, output)

    axis = range(10)
    visualize(code, axis, "./img/pca/" + code_path + "_pca_{}".format(axis))

    axis = [(0, 1), (8, 9)]
    for ax in axis:
        visualize(code, axis, "./img/pca/" + code_path + "_pca_{}".format(axis))
