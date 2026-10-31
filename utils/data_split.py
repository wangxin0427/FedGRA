import torch
from collections import defaultdict
from torch.utils.data import Subset
import numpy as np

def split_dataset_noniid(dataset, num_clients, alpha=0.5):
    # ★ 修复：转成 int
    labels = [int(dataset[i][1]) for i in range(len(dataset))]

    if alpha is None or (isinstance(alpha, str) and alpha.lower() == "iid"):
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        client_indices = np.array_split(indices, num_clients)
        client_indices = [idx.tolist() for idx in client_indices]

    else:
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)

        client_indices = [[] for _ in range(num_clients)]

        for label, indices in label_to_indices.items():
            indices = np.random.permutation(indices).tolist()
            n = len(indices)

            proportions = torch.distributions.Dirichlet(
                torch.full((num_clients,), float(alpha))
            ).sample()
            proportions = proportions / proportions.sum()

            # floor 分配，保证至少 0（后面全局兜底）
            counts = (proportions * n).floor().long()
            remainder = n - counts.sum().item()

            # 把余数分给 proportion 最大的客户端
            if remainder > 0:
                _, sorted_idx = proportions.sort(descending=True)
                for i in range(remainder):
                    counts[sorted_idx[i % num_clients]] += 1

            assert counts.sum().item() == n, f"类别 {label} 分配总数不对"

            start = 0
            for cid in range(num_clients):
                end = start + counts[cid].item()
                client_indices[cid].extend(indices[start:end])
                start = end

    # ★ 全局兜底：从所有客户端均匀借，而不是只薅 richest
    min_required = 64  # 至少够一个 mini-batch
    for cid in range(num_clients):
        while len(client_indices[cid]) < min_required:
            # 从所有够多的客户端各借 1 个
            donors = [c for c in range(num_clients) 
                      if c != cid and len(client_indices[c]) > min_required + 1]
            if not donors:
                break  # 实在借不到就算了（样本太少没办法）
            donor = np.random.choice(donors)
            client_indices[cid].append(client_indices[donor].pop())

    # 最终检查
    empty = sum(1 for idxs in client_indices if len(idxs) == 0)
    sizes = [len(idxs) for idxs in client_indices]
    print(f"[NonIID Split] α={alpha}, clients={num_clients}, "
          f"sizes: min={min(sizes)}, max={max(sizes)}, empty={empty}")

    client_datasets = [Subset(dataset, idxs) for idxs in client_indices]
    client_sizes = {i: len(idxs) for i, idxs in enumerate(client_indices)}
    return client_datasets, client_sizes