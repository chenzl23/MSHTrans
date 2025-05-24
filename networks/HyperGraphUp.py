import torch


def upHyperGraph(H, kernel_size, stride, target_n, device):
    ori_n = H.size(0)
    ori_m = H.size(1)
    target_m = ori_m
    up_H = torch.zeros((target_n, target_m)).to(device)
    for i in range(ori_n):
        start_idx = i * stride
        end_idx = start_idx + kernel_size
        up_H[start_idx:end_idx] += H[i]
    return up_H

def HypergraphFusion(H_list, kernel_size_list, stride_list, target_n, device):
    up_kernel_size_list = torch.ones(len(kernel_size_list), dtype=torch.int32).to(device)
    up_kernel_size_list[0] = kernel_size_list[0]
    
    fused_H = H_list[0]
    full_H_list = []
    
    for i in range(1, len(kernel_size_list)):
        up_kernel_size_list[i] = up_kernel_size_list[i - 1] * kernel_size_list[i]
    for i in range(len(H_list) - 1, 0, -1):
        up_H = upHyperGraph(H_list[i], up_kernel_size_list[i - 1], stride_list[i - 1], target_n, device)
        fused_H = torch.concatenate([fused_H, up_H], dim=1)
        full_H_list.append(up_H)
    full_H_list.append(H_list[0])
    return fused_H, full_H_list


