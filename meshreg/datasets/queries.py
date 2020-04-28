from enum import Enum, auto


class BaseQueries(Enum):
    CAMINTR = auto()
    OBJFACES = auto()
    OBJCORNERS2D = auto()
    OBJCORNERS3D = auto()
    OBJVERTS3D = auto()
    OBJVERTS2D = auto()
    OBJVIS2D = auto()
    HANDVERTS3D = auto()
    HANDVERTS2D = auto()
    HANDVIS2D = auto()
    JOINTS3D = auto()
    JOINTS2D = auto()
    IMAGE = auto()
    SIDE = auto()
    OBJCANVERTS = auto()
    OBJCANROTVERTS = auto()
    OBJCANROTCORNERS = auto()
    OBJCANSCALE = auto()
    OBJCANTRANS = auto()
    OBJCANCORNERS = auto()
    JOINTVIS = auto()


class TransQueries(Enum):
    CAMINTR = auto()
    OBJVERTS3D = auto()
    OBJVERTS2D = auto()
    OBJCORNERS2D = auto()
    OBJCORNERS3D = auto()
    OBJCANROTVERTS = auto()
    OBJCANROTCORNERS = auto()
    HANDVERTS3D = auto()
    HANDVERTS2D = auto()
    JOINTS3D = auto()
    JOINTS2D = auto()
    CENTER3D = auto()
    IMAGE = auto()
    JITTERMASK = auto()
    SIDE = auto()
    SCALE = auto()
    AFFINETRANS = auto()
    ROTMAT = auto()


def one_query_in(candidate_queries, base_queries):
    for query in candidate_queries:
        if query in base_queries:
            return True
    return False


def get_trans_queries(base_queries):
    trans_queries = []
    if BaseQueries.OBJVERTS3D in base_queries:
        trans_queries.append(TransQueries.OBJVERTS3D)
    if BaseQueries.IMAGE in base_queries:
        trans_queries.append(TransQueries.IMAGE)
        trans_queries.append(TransQueries.AFFINETRANS)
        trans_queries.append(TransQueries.ROTMAT)
        trans_queries.append(TransQueries.JITTERMASK)
    if BaseQueries.JOINTS2D in base_queries:
        trans_queries.append(TransQueries.JOINTS2D)
    if BaseQueries.JOINTS3D in base_queries:
        trans_queries.append(TransQueries.JOINTS3D)
    if BaseQueries.HANDVERTS3D in base_queries:
        trans_queries.append(TransQueries.HANDVERTS3D)
        trans_queries.append(TransQueries.CENTER3D)
    if BaseQueries.HANDVERTS2D in base_queries:
        trans_queries.append(TransQueries.HANDVERTS2D)
    if BaseQueries.OBJVERTS3D in base_queries:
        trans_queries.append(TransQueries.OBJVERTS3D)
    if BaseQueries.OBJVERTS2D in base_queries:
        trans_queries.append(TransQueries.OBJVERTS2D)
    if BaseQueries.OBJCORNERS3D in base_queries:
        trans_queries.append(TransQueries.OBJCORNERS3D)
    if BaseQueries.OBJCORNERS2D in base_queries:
        trans_queries.append(TransQueries.OBJCORNERS2D)
    if BaseQueries.OBJCANROTCORNERS in base_queries:
        trans_queries.append(TransQueries.OBJCANROTCORNERS)
    if BaseQueries.OBJCANROTVERTS in base_queries:
        trans_queries.append(TransQueries.OBJCANROTVERTS)
    if BaseQueries.CAMINTR in base_queries:
        trans_queries.append(TransQueries.CAMINTR)
    if BaseQueries.OBJCANVERTS in base_queries or BaseQueries.OBJCANCORNERS:
        trans_queries.append(BaseQueries.OBJCANSCALE)
        trans_queries.append(BaseQueries.OBJCANTRANS)
    return trans_queries
