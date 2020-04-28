import numpy as np
import numpy.matlib


def get_rigid_transform(A, B):
    cenA = np.mean(A, 0)  # 3
    cenB = np.mean(B, 0)  # 3
    N = A.shape[0]  # 24
    H = np.dot((B - np.matlib.repmat(cenB, N, 1)).transpose(), (A - np.matlib.repmat(cenA, N, 1)))

    [U, _, V] = np.linalg.svd(H)
    R = np.dot(U, V)  # matlab returns the transpose: .transpose()
    if np.linalg.det(R) < 0:
        U[:, 2] = -U[:, 2]
        R = np.dot(U, V.transpose())
    t = np.dot(-R, cenA.transpose()) + cenB.transpose()
    return R, t


def get_affine_trans(target, source):
    rigid_transform = get_rigid_transform(source, target)

    # Concatenate rotation and translation
    rigid_transform = np.asarray(
        np.concatenate((rigid_transform[0], np.matrix(rigid_transform[1]).T), axis=1)
    )
    rigid_transform = np.concatenate((rigid_transform, np.array([[0, 0, 0, 1]])))
    return rigid_transform
