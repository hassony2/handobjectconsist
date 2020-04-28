import os

import numpy as np
from libyana.meshutils import meshio


def load_objects(obj_root):
    object_names = [obj_name for obj_name in os.listdir(obj_root) if ".tgz" not in obj_name]
    objects = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, obj_name, "textured_simple_2000.obj")
        with open(obj_path) as m_f:
            mesh = meshio.fast_load_obj(m_f)[0]
        objects[obj_name] = {"verts": mesh["vertices"], "faces": mesh["faces"]}
    return objects


def load_corners(corner_root):
    obj_corners = {}
    for objname in os.listdir(corner_root):
        filepath = os.path.join(corner_root, objname, "corners.npy")
        corners = np.load(filepath)
        obj_corners[objname] = corners
    return obj_corners


def lineParser(line, annoDict):
    """
    Parses a line in the 'anno.txt' and creates a entry in dict with lineid as key
    :param line: line from 'anno.txt'
    :param annoDict: dict in which an entry should be added
    :return:
    """
    lineList = line.split()
    lineid = lineList[0]
    objID = lineList[1]
    paramsList = list(map(float, lineList[2:]))

    assert lineid not in annoDict.keys(), "Something wrong with the annotation file..."

    annoDict[lineid] = {
        "objID": objID,
        "handJoints": np.reshape(np.array(paramsList[:63]), [21, 3]),
        "handPose": np.array(paramsList[63 : 63 + 48]),
        "handTrans": np.array(paramsList[63 + 48 : 63 + 48 + 3]),
        "handBeta": np.array(paramsList[63 + 48 + 3 : 63 + 48 + 3 + 10]),
        "objRot": np.array(paramsList[63 + 48 + 3 + 10 : 63 + 48 + 3 + 10 + 3]),
        "objTrans": np.array(paramsList[63 + 48 + 3 + 10 + 3 : 63 + 48 + 3 + 10 + 3 + 3]),
    }
    return annoDict


def parseAnnoTxt(filename):
    """
    Parse the 'anno.txt'
    :param filename: path to 'anno.txt'
    :return: dict with lineid as keys
    """
    ftxt = open(filename, "r")
    annoLines = ftxt.readlines()
    annoDict = {}
    for line in annoLines:
        lineParser(line, annoDict)

    return annoDict


def project3DPoints(camMat, pts3D, isOpenGLCoords=True):
    """
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis.
        If False hand/object along positive z-axis
    :return:
    """
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coordChangeMat = np.array([[1.0, 0.0, 0.0], [0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    if isOpenGLCoords:
        pts3D = pts3D.dot(coordChangeMat.T)

    projPts = pts3D.dot(camMat.T)
    projPts = np.stack([projPts[:, 0] / projPts[:, 2], projPts[:, 1] / projPts[:, 2]], axis=1)

    assert len(projPts.shape) == 2

    return projPts
