import random, numpy as np, pandas as pd, torch
from omegaconf import OmegaConf, DictConfig
from typing import List


def set_seed(seed: int) -> None:
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_config(conf: DictConfig) -> dict:
    [dict] = pd.json_normalize(
        OmegaConf.to_container(conf, resolve=True), sep="."
    ).to_dict(orient="records")
    return dict


def pad_sequence(seq: torch.Tensor, pad: int) -> torch.Tensor:
    """Pads a sequence with random one-hot encoded vector."""
    if seq.ndim == 2:
        t, d = seq.shape
        pad_tensor = torch.zeros((pad, d), device=seq.device)
        random_indices = torch.randint(0, d, (pad,), device=seq.device).unsqueeze(1)
        pad_tensor.scatter_(dim=-1, index=random_indices, value=1.0)
        return torch.cat((pad_tensor, seq), dim=0)

    b, _, d = seq.shape
    pad_tensor = torch.zeros((b, pad, d), device=seq.device)
    random_indices = torch.randint(0, d, (b, pad), device=seq.device).unsqueeze(1)
    pad_tensor.scatter_(dim=-1, index=random_indices, value=1.0)
    return torch.cat((pad_tensor, seq), dim=1)


def split_into_windows(
    seq: torch.Tensor, window: int, pad: int = 0, vectorize: bool = False
) -> torch.Tensor:
    """Splits a sequence into windows of length window."""
    if pad > 0:
        seq = pad_sequence(seq, pad)

    if vectorize:
        _, _, dim = seq.shape
        xs = seq.unfold(dimension=1, size=window, step=1)
        xs = xs[:, :-1].transpose(-1, -2)
        ys = seq[:, window:]
        # flatten the batch and sequence dimensions
        xs = xs.contiguous().view(-1, window, dim)
        ys = ys.contiguous().view(-1, dim)
        return xs, ys

    else:
        return torch.flatten(
            torch.stack(
                [seq[:, i : i + window, :] for i in range(seq.shape[1] - window)], dim=1
            ),
            0,
            1,
        ), torch.flatten(
            torch.stack([seq[:, i, :] for i in range(window, seq.shape[1])], dim=1),
            0,
            1,
        )


def random_unit_norm_matrix(
    d: int, rank: int = -1, rectangular: bool = False
) -> torch.Tensor:
    """Returns a random matrix with unit norm."""
    if rank == -1:
        rank = d

    A = torch.randn(d, d)
    U, S, V = torch.linalg.svd(A)

    if rectangular:  # return a rectangular matrix
        return U[:, :rank]

    S[:rank] = 1
    S[rank:] = 0
    return U @ torch.diag(S) @ V.T


def random_orthogonal_matrices(
    n: int, d: int, rank: int = -1
) -> list[torch.Tensor]:
    """Returns n random d×d matrices that are mutually Frobenius-orthogonal.

    Orthogonality means Tr(A_i^T A_j) = 0 for i ≠ j.
    Each matrix preserves the Frobenius norm of a random_unit_norm_matrix.
    """
    assert n <= d * d, f"Cannot generate {n} orthogonal matrices in R^{d}x{d} (max {d*d})"

    matrices = [random_unit_norm_matrix(d, rank) for _ in range(n)]
    target_norm = matrices[0].norm()  # Frobenius norm to restore after orthogonalization

    # Flatten to vectors in R^{d²} and apply Gram-Schmidt
    vecs = [m.flatten() for m in matrices]
    ortho_vecs = []
    for v in vecs:
        for u in ortho_vecs:
            v = v - torch.dot(v, u) * u
        norm = v.norm()
        if norm > 1e-10:
            v = v / norm
        ortho_vecs.append(v)

    return [v.reshape(d, d) * target_norm for v in ortho_vecs]