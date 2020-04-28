import torch

from manopth import manolayer


def get_closed_faces():
    mano_layer = manolayer.ManoLayer(
        joint_rot_mode="axisang", use_pca=False, mano_root="assets/mano", center_idx=None, flat_hand_mean=True
    )
    close_faces = torch.Tensor(
        [
            [92, 38, 122],
            [234, 92, 122],
            [239, 234, 122],
            [279, 239, 122],
            [215, 279, 122],
            [215, 122, 118],
            [215, 118, 117],
            [215, 117, 119],
            [215, 119, 120],
            [215, 120, 108],
            [215, 108, 79],
            [215, 79, 78],
            [215, 78, 121],
            [214, 215, 121],
        ]
    )
    closed_faces = torch.cat([mano_layer.th_faces, close_faces.long()])
    # Indices of faces added during closing --> should be ignored as they match the wrist
    # part of the hand, which is not an external surface of the human

    # Valid because added closed faces are at the end
    hand_ignore_faces = [1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551]

    return closed_faces, hand_ignore_faces
