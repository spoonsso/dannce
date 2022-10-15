import numpy as np

def matlab_pose_to_cv2_pose(camParamsOrig):
    keys = ['K', 'RDistort', 'TDistort', 't', 'r']
    camParams = list()
    for icam in range(len(camParamsOrig)):
        camParamOrig = camParamsOrig[icam]
        camParam = camParamOrig.copy()
        if 'R' not in camParam: camParam['R'] = camParamOrig['r']
        camParams.append(camParam)

    # from matlab to opencv
    ba_poses = dict()
    for icam in range(len(camParams)):
        rd = camParams[icam]['RDistort'].reshape((-1))
        td = camParams[icam]['TDistort'].reshape((-1))
        dist = np.zeros((8,))
        dist[:2] = rd[:2]
        dist[2:4] = td[:2]
        dist[4]  = rd[2]
        ba_poses[icam] = {'R': camParams[icam]['R'].T, 
                    't': camParams[icam]['t'].reshape((3,)),
                    'K': camParams[icam]['K'].T - [[0, 0, 1], [0,0,1], [0,0,0]],
                    'dist':dist}
    return ba_poses


def cv2_pose_to_matlab_pose(ba_poses):
    camParams = list()
    for icam in range(len(ba_poses)):
        camParam = dict()
        camParam['K'] = (np.array(ba_poses[icam]['K']) + [[0, 0, 1], [0,0,1], [0,0,0]]).T
        camParam['R'] = np.array(ba_poses[icam]['R']).T
        camParam['t'] = np.array(ba_poses[icam]['t']).reshape((1,3))
        dist = ba_poses[icam]['dist']
        rd = np.zeros((3,))
        rd[:2] = dist[:2]
        rd[2]  = dist[4]
        td = np.array(ba_poses[icam]['dist'][2:4])
        camParam['RDistort'] = rd.reshape((1,-1))
        camParam['TDistort'] = td.reshape((1,-1))
        camParams.append(camParam)
    return camParams
