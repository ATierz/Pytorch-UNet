import os
import torch
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt



folder_path = r'data/points'
pt_name = 'beam_35_10_20_2000_27.5_7.5_7.5_0_test'

real_points = torch.load(os.path.join(folder_path, 'position_points_p6_flip_f8.pt'))
synthetic_points = torch.load(os.path.join(folder_path, pt_name + '.pt'))
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_xlabel('X'), ax.set_ylabel('Y')
# for i in range(6):
#
#     ax.scatter(real_points[i, :, 0], real_points[i, :, 1], marker='o')
#     # ax.scatter(x_2[:, 0], x_2[:, 2], x_2[:, 1], marker='o')
#     # ax.scatter(pos_ini[:, 0], pos_ini[:, 2], pos_ini[:, 1], marker='o')
# plt.show()

beam_x, beam_y, beam_z = 35, 10, 20
mesh = 5
data_mix = []
pos_real_list = []
for i, data_syn in enumerate(synthetic_points):
    points_syn = data_syn.x
    pos = real_points[i]
    if i == 0:
        min_x = min(pos[:, 0]).clone()
        min_y = min(pos[:, 1]).clone()
        max_x = max(pos[:, 0]).clone()



    pos[:,0] = pos[:, 0] - min_x
    pos[:,1] = pos[:, 1] - min_y
    if i == 0:
        max_y = max(pos[:, 1]).clone()
    pos[:,1] = -1*(pos[:,1].clone() - max_y)


    # x_max_px, y_max_px = max(pos[:,0]), max(pos[:,1])
    if i == 0:
        ratio_px = (beam_x / int(max(pos[:, 0])) + beam_y / int(max(pos[:, 1]))) / 2
    # print(beam_x, int(max_x), beam_y, int(max_y), ratio_px)
    num_rep_z = int((beam_z / mesh)) +1


    # Creamos una nueva columna con 0s para los primeros 24 puntos y 1s para los siguientes 24 puntos
    z_dim = torch.zeros(pos.shape[0])
    for j in range(num_rep_z - 1):
        z_dim = torch.cat([z_dim, torch.ones(pos.shape[0]) * (j + 1) * mesh])
    pos = (pos * ratio_px).repeat(num_rep_z, 1)
    pos_real = torch.cat([pos, z_dim.unsqueeze(1)], dim=1)/100
    # pos_real[:,1] = -1*(pos_real[:,1].clone() - torch.max(pos_real[:,1]))
    # Crear un árbol KD a partir de los puntos de B
    tree = cKDTree(pos_real)

    # Para cada punto en A, encontrar el punto más cercano en B
    distances, indices = tree.query(points_syn[:, 0:3])
    print(np.round(max(distances), 4), indices[:16])
    pos_real_list.append(pos_real[indices])
    if i == 0:
        pos_ini = np.asarray(pos_real[indices]).copy()
    x_ = np.asarray(pos_real[indices])
    x_2 = points_syn[:, 0:3]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X'), ax.set_ylabel('Y')
    ax.scatter(x_[:, 0], x_[:, 2], x_[:, 1], marker='o')
    ax.scatter(x_2[:, 0], x_2[:, 2], x_2[:, 1], marker='o')
    # ax.scatter(pos_ini[:, 0], pos_ini[:, 2], pos_ini[:, 1], marker='o')
    plt.show()

for i in range(len(pos_real_list)-1):

    synthetic_points[i].x[:, 0:3] = pos_real_list[i]
    synthetic_points[i].y[:, 0:3] = pos_real_list[i+1]

    data_mix.append(synthetic_points[i])

torch.save(data_mix, os.path.join(folder_path,  pt_name + 'mix.pt'))


def claculate_match_points(points_syn, pos_real, dInfo):

    pos_real[:,0] = pos_real[:, 0] - min(pos_real[:, 0])
    pos_real[:,1] = pos_real[:, 1] - min(pos_real[:, 1])
    x_max_px, y_max_px = max(pos_real[:,0]), max(pos_real[:,1])
    ratio_px = (dInfo['dim']['length']/int(x_max_px) + dInfo['dim']['high']/int(y_max_px))/2

    num_rep_z = int((dInfo['dim']['width'] / dInfo['dim']['mesh'])) +1


    # Creamos una nueva columna con 0s para los primeros 24 puntos y 1s para los siguientes 24 puntos
    z_dim = torch.zeros(pos_real.shape[0])
    for i in range(num_rep_z - 1):
        z_dim = torch.cat([z_dim, torch.ones(pos_real.shape[0]) * (i + 1) * dInfo['dim']['mesh']])
    pos = (pos_real * ratio_px).repeat(num_rep_z, 1)
    pos_real = torch.cat([pos, z_dim.unsqueeze(1)], dim=1)/100

    # Crear un árbol KD a partir de los puntos de B
    tree = cKDTree(points_syn[:,0:3])

    # Para cada punto en A, encontrar el punto más cercano en B
    distances, indices = tree.query(pos_real)
    return distances, indices