import numpy as np
import torch

from libyana.flowutils import dispflow


def colvis_pairs(inps1, inps2, row_idx):
    outs = []
    for inp1, inp2 in zip(inps1, inps2):
        outs.append(inp1[row_idx])
        outs.append(inp2[row_idx])
    out = torch.cat(outs, 0)
    return out


def colvis_pairef(inps, ref, row_idx):
    row_ref = ref[row_idx]
    outs = []
    for inp in inps:
        outs.append(row_ref)
        outs.append(inp[row_idx])
    out = torch.cat(outs, 0)
    return out


def colvis_flowpairs(flows1, flows2, row_idx):
    all_flows = []
    for flow1, flow2 in zip(flows1, flows2):
        flow_s = dispflow.disp_flow(flow1[row_idx])
        flow_e = dispflow.disp_flow(flow2[row_idx])
        all_flows.append(flow_s)
        all_flows.append(flow_e)
    show_flows = np.concatenate(all_flows, 0)
    return show_flows
