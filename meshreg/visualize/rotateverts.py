from manopth import rodrigues_layer


def rotate_verts(verts, axisang=(0, 1, 1)):
    centroids = verts.mean(1).unsqueeze(1)
    verts_c = verts - centroids
    rot_mats = rodrigues_layer.batch_rodrigues(verts.new(axisang).unsqueeze(0)).view(1, 3, 3)
    verts_cr = rot_mats.repeat(verts.shape[0], 1, 1).bmm(verts_c.transpose(1, 2)).transpose(1, 2)
    verts_final = verts_cr + centroids
    return verts_final
