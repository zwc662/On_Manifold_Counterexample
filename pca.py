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


def visualize(code, axis = None, path = None):
    rows = len(code.keys())
    if axis is not None:
        cols = len(axis)
    else:
        cols = len(code[0])
        axis = range(cols) 
    print(np.asarray(axis))
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

    fig.subplots_adjust(hspace=1., wspace = 0.5)
    if path is not None:
        path = "./img/pca/{}_{}".format(axis, timestamp)
    plt.savefig(path)
    plt.show()

if __name__ == '__main__':
    code_path = "code_dict_latent_10_tot_100000_2020_07_12_16_40_32"
    code = pca("./data/" + code_path, 10)
    axis = (0, 1)

    visualize(code, [axis], "./img/pca/" + code_path + "_pca_{}".format(axis))
    dump_pca_code(code, axis, ("./data/" + code_path + "_pca_{}".format(axis), "wb"))
