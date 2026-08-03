import sys
import time
from pathlib import Path

import numpy as np
import torch
from ldpc import bposd_decoder

from .args import args

from gnd_decoder.paths import PROJECT_ROOT, ARTIFACTS_DIR, resolve_path, resolve_output_path
from gnd_decoder.core import Abstractcode, Errormodel, Loading_code, mod2, read_code  # noqa: E402


def build_error_rates():
    if not args.sweep:
        return torch.tensor([args.er], dtype=dtype)

    er_min = args.er_min if args.er_min is not None else 10 ** (-3)
    er_max = args.er_max if args.er_max is not None else 10 ** (-1)
    return torch.logspace(np.log10(er_min), np.log10(er_max), args.n_points)


def safe_std(values):
    if len(values) < 2:
        return 0.0
    return torch.tensor(values).std().item()

n, d, k, seed, c_type = args.n, args.d, args.k, args.seed, args.c_type# 90, 4, 8, 0, 'qcc'
trials = args.trials
device, dtype = 'cpu', torch.float64
mod2 = mod2(device=device, dtype=dtype)

if c_type == 'drsur':
    defect_g = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]#[14, 16, 18]
    info = read_code(d=7, k=1, seed=seed, c_type='rsur', n=n)
    oCode = Loading_code(info)
    g = oCode.g_stabilizer[defect_g,:]
    code = Abstractcode(g_stabilizer=g)
else:    
    info = read_code(d=d, k=k, seed=seed, c_type=c_type, n=n)
    code = Loading_code(info)
    
n = code.n
PCM = code.PCM.cpu().numpy()

l1 = mod2.rep(code.logical_opt).int().numpy()
l = np.zeros_like(l1)
l[:, :n], l[:, n:] = l1[:, n:], l1[:, :n]
# print(l)

L = []
error_rate = build_error_rates()
tt = torch.zeros(len(error_rate))
for i in range(len(error_rate)):
    er_value = float(error_rate[i])
    E = Errormodel(e_rate=er_value)

    bpd=bposd_decoder(
    PCM,#the parity check matrix
    error_rate=2 * er_value,
    channel_probs=[None], #assign error_rate to each qubit. This will override "error_rate" input variable
    max_iter=1000, #the maximum number of iterations for BP)
    bp_method="ms",
    ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
    osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
    osd_order=7 #the osd search depth
    )

    
    correct_number = 0
    t = 0
    processed = 0
    batch_seed = int(10000 * float(error_rate[i]))
    while processed < trials:
        batch_size = min(args.chunk_size, trials - processed)
        error = E.generate_error(n=code.n, m=batch_size, seed=batch_seed + processed)
        if error.dim() == 1:
            error = error.unsqueeze(0)
        syndrome = mod2.commute(error, code.g_stabilizer)
        if syndrome.dim() == 1:
            syndrome = syndrome.unsqueeze(0)
        error = mod2.rep(error).int().numpy()
        syndrome = syndrome.numpy()

        for x in range(batch_size):
            e = error[x]
            s = syndrome[x]

            t1 = time.time()
            bpd.decode(s)
            recover = bpd.osdw_decoding
            check = (e + recover)%2
            s = np.sum((check @ l.T) %2)
            t2 = time.time()
            t = t+(t2-t1)
            if s == 0:
                correct_number+=1
        processed += batch_size

    lorate = 1 - correct_number/trials
    ta = t/trials
    print(lorate)
    print(ta)
    tt[i] = ta
    L.append(lorate)
print(L)
print(tt.mean().item(), safe_std(tt.tolist()))
        # print('Error:')
        # print(e)
        # print('Decoding:')
        # print(recover)
        # print('Check:')
        # print(check)
