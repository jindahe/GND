import sys
import time
from pathlib import Path
import math

import numpy as np
import torch
from pymatching import Matching

from args import args

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module import Abstractcode, Errormodel, Loading_code, mod2, read_code  # noqa: E402


def build_error_rates():
    if not args.sweep:
        return torch.tensor([args.er], dtype=dtype)

    er_min = args.er_min if args.er_min is not None else 10 ** (-1.5)
    er_max = args.er_max if args.er_max is not None else 10 ** (-0.5)
    return torch.logspace(np.log10(er_min), np.log10(er_max), args.n_points)


def safe_std(values):
    if len(values) < 2:
        return 0.0
    return torch.tensor(values).std().item()

n, d, k, seed, c_type = args.n, args.d, args.k, args.seed, args.c_type
trials = args.trials
device, dtype = 'cpu', torch.float32
e_model = args.e_model
error_seed = args.error_seed
mod2 = mod2(device=device, dtype=dtype)
if c_type == 'drsur':
    defect_g = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]#[14, 16, 18]
    info = read_code(d=7, k=1, n=n, seed=seed, c_type='rsur')
    oCode = Loading_code(info)
    g = oCode.g_stabilizer[defect_g,:]
    code = Abstractcode(g_stabilizer=g)
else:    
    info = read_code(n=n, d=d, k=k, seed=seed, c_type=c_type)
    code = Loading_code(info)
n = code.n
PCM = code.PCM.cpu().numpy()

l1 = mod2.rep(code.logical_opt).int().numpy()
l = np.zeros_like(l1)
l[:, :n], l[:, n:] = l1[:, n:], l1[:, :n]
#print(l)



    
L = []
error_rate = build_error_rates()
print(error_rate)
tt = torch.zeros(len(error_rate))
for i in range(len(error_rate)):
    '''generate error'''
    er_value = float(error_rate[i])
    E = Errormodel(er_value, e_model=e_model)
    errors = E.generate_error(code.n,  m=trials, seed=error_seed)
    if errors.dim() == 1:
        errors = errors.unsqueeze(0)
    if e_model == 'dep':
        er = 2 * er_value / 3
    elif e_model == 'dep2':
        er = 8 * er_value / 15

    weights = torch.ones(2 * code.n) * math.log((1 - er) / er)
    Decoder = Matching(PCM, weights=weights)
    syndrome = mod2.commute(errors, code.g_stabilizer)
    if syndrome.dim() == 1:
        syndrome = syndrome.unsqueeze(0)
    error = mod2.rep(errors).int().numpy()
    syndrome = syndrome.numpy()

    correct_number = 0
    t = 0
    for j in range(trials):
        e = error[j]
        s = syndrome[j]

        t1 = time.time()
        #print(s)
        recover = Decoder.decode(s)
        check = (e + recover)%2
        s = np.sum((check @ l.T) %2)
        t2 = time.time()
        t = t+(t2-t1)
        if s == 0:
            correct_number+=1
        
    lorate = 1 - correct_number/trials
    ta = t#/trials
    print(lorate)
    print(ta)
    tt[i] = ta
    L.append(int(10000*lorate)/10000)
print(L)
print(tt.mean().item(), safe_std(tt.tolist()))
