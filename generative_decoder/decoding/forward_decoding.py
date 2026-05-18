import sys
import time
from pathlib import Path

import torch

from args import args

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module import Errormodel, Loading_code, MADE, NADE, TraDE_binary, mod2, read_code  # noqa: E402


def safe_std(values):
    if len(values) < 2:
        return 0.0
    return torch.tensor(values).std().item()

def forward(n_s, m, van, syndrome, device, dtype, k=1, n_type='made'):
    if n_type == 'made':
        condition = syndrome*2-1
        x = (van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k) + 1)/2
    elif n_type in ('trade', 'nade'):
        condition = syndrome
        x = van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k)
    x = x[:, m:m+2*k]
    return x

'''Hyper Paras:'''
if args.dtype == 'float32':
    dtype = torch.float32
elif args.dtype == 'float64':
    dtype = torch.float64
device = args.device
trials = args.trials
'''Code Paras:'''
c_type, n, d, k, seed = args.c_type, args.n, args.d, args.k, args.seed
'''Net Paras:'''
l_type, n_type, e_model = 'fkl', args.n_type, args.e_model
er = args.er
'''seed for sampling errors:'''
error_seed = args.error_seed
path = PROJECT_ROOT / "net" / "code_capacity" / f"{n_type}_{c_type}_n{n}_d{d}_k{k}_seed{seed}_er{er}_{e_model}.pt"
print(path)

error_rate = torch.tensor([er], dtype=dtype)
print(error_rate)
'''Loading Code'''
info = read_code(d, k, n=n, seed=seed, c_type=c_type)
Code = Loading_code(info)

'''Loading Net'''
net = torch.load(path, map_location=device, weights_only=False)

if n_type == 'made':
    van = MADE(Code.m+2*k, 0, 1).to(device).to(dtype)
    van.deep_net = net.to(device).to(dtype)
elif n_type == 'trade':
    van = net.to(device).to(dtype)
    van.device = device
    van.dtype = dtype
elif n_type == 'nade':
    van = net.to(device).to(dtype)

mod2 = mod2(device=device, dtype=dtype)




t = []
lo_rate = []
std = []
for i in range(len(error_rate)):
    if trials <= 10000:
        '''generate error'''

        E = Errormodel(error_rate[i], e_model=e_model)#error_rate[i]
        errors = E.generate_error(Code.n,  m=trials, seed=error_seed)

        '''getting syndrome'''
        syndrome = mod2.commute(errors, Code.g_stabilizer)
        # print(errors[errors.nonzero()[0, 0]])
        pe = E.pure(Code.pure_es, syndrome, device=device, dtype=dtype) #getting pure error from the syndrome

        '''forward to get configs'''
        t0 = time.time()
        lconf = forward(n_s=trials, m=Code.m, van=van, syndrome=syndrome, device=device,dtype=dtype, k=k, n_type=n_type) 


        '''correction'''
        l = mod2.confs_to_opt(confs=lconf, gs=Code.logical_opt) # obtain logical operator
        recover = mod2.opt_prod(pe, l) # obtain recover operator
        check = mod2.opt_prod(recover, errors) # obtain check operator
        commute = mod2.commute(check, Code.logical_opt) # check the commutation relation
        # print( torch.count_nonzero(commute))
        if trials == 1:
            fail = torch.count_nonzero(commute.sum(0))
        else:
            fail = torch.count_nonzero(commute.sum(1))
        t1=time.time()
        logical_error_rate = fail/trials
        print(logical_error_rate)
    #         if commute == 0:
    #             correct_number+=1
    #         print(correct_number, j+1)    
    #     logical_error_rate = 1-correct_number/trials
    #     print(logical_error_rate)
        
    else:
        ns = 100000
        a = int(trials/(ns*10))
        logical_error_rate = []
        for ii in range(10):
            ler = 0
            for j in range(a):
                print(i, ii, j)
                E = Errormodel(error_rate[i], e_model=e_model)#error_rate[i]
                errors = E.generate_error(Code.n,  m=ns, seed=int(1000*error_rate[i])+100*j+285*ii)
                syndrome = mod2.commute(errors, Code.g_stabilizer)

                pe = E.pure(Code.pure_es, syndrome, device=device, dtype=dtype)

                '''forward to get configs'''
                t0 = time.time()
                lconf = forward(n_s=ns, m=Code.m, van=van, syndrome=syndrome, device=device,dtype=dtype, k=k, n_type=n_type)      
                '''correction'''
                l = mod2.confs_to_opt(confs=lconf, gs=Code.logical_opt)
                recover = mod2.opt_prod(pe, l)
                check = mod2.opt_prod(recover, errors)
                commute = mod2.commute(check, Code.logical_opt)
                if trials == 1:
                    fail = torch.count_nonzero(commute.sum(0))
                else:
                    fail = torch.count_nonzero(commute.sum(1))
                t1=time.time()
                lr = fail/ns
                print(lr)
                ler += lr.cpu().item()
            logical_error_rate.append(ler/a)
        std.append(torch.tensor(logical_error_rate).std().cpu().item())
        print('std:',torch.tensor(logical_error_rate).std())
        logical_error_rate = torch.tensor(logical_error_rate).mean()

        print(logical_error_rate)
    t.append(t1-t0)
    lo_rate.append(logical_error_rate.cpu().item())
print(lo_rate)
print(std)
print(torch.tensor(t).mean().item(), safe_std(t))
