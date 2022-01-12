import argparse
import logging
import subprocess
import time
from typing import List, Union

import torch


def main(int_or_list_gpu:Union[int, List[int]]) -> None:
    if int_or_list_gpu == -1:
        list_gpu = list(range(torch.cuda.device_count()))
    else:
        list_gpu = int_or_list_gpu
    
    list_tensor = []
    tensor_1gb = torch.randint(1000, (1000, 131000))
    for gpu_id in list_gpu:
        device = torch.device(f'cuda:{gpu_id}')
        # allocate torch gpu space
        list_used_origin = memory_nvidiasmi('used')
        _ = torch.tensor(0).to(device)
        del _
        torch.cuda.empty_cache()
        list_used_init = memory_nvidiasmi('used')
        list_changed_idx = [i for i, (or_mib, in_mib) in enumerate(zip(list_used_origin, list_used_init)) if or_mib < in_mib]
        assert len(list_changed_idx) == 1, 'Something wrong in allocate 0 tensor'

        nvidiasmi_id = list_changed_idx[0]
        # calculate free memory
        list_free = memory_nvidiasmi('free')
        target_mib = list_free[nvidiasmi_id] - list_used_init[nvidiasmi_id]
        for _ in range(int(target_mib // 1000)):
            list_tensor.append(tensor_1gb.to(device))
        allocated_mib = memory_nvidiasmi('used')[nvidiasmi_id]
        total_memory = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        logger.info(f'[GPU{gpu_id}] {allocated_mib}MiB/{total_memory}MiB is allocated')
    logger.info('All GPU is allocated, so enter to infinite loop')
    while True:
        time.sleep(100000)
        

def memory_nvidiasmi(mode:str) -> List[int]:
    assert mode in ['free', 'used'], 'mode must be "free" or "used"'
    o = subprocess.run(
        ['nvidia-smi', f'--query-gpu=memory.{mode}', '--format=csv,noheader'], 
        stdout=subprocess.PIPE, 
        encoding='utf-8'
    )
    list_mib = [int(s.replace(' MiB', '')) for s in o.stdout.split('\n') if ' MiB' in s]
    return list_mib


def default_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s: %(message)s', 
        datefmt='%Y/%m/%d %H:%M:%S'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def read_args():
    parser = argparse.ArgumentParser(description='Occupy GPU memory almost full')
    parser.add_argument('-g', '--gpu', type=int, default=-1, nargs='*', help='Indices of gpus to occupy. default:-1 (all gpus)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = read_args()
    logger = default_logger()
    main(args.gpu)