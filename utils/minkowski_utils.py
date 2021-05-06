import torch

def ME_sparse_to_dense(x, shape=None, min_coordinate=None, contract_stride=True):
    """
    rewrite ME's function https://github.com/NVIDIA/MinkowskiEngine/blob/7ccf01b392c5444ab155cdb024c35a84faae20aa/MinkowskiEngine/MinkowskiSparseTensor.py#L486
    due to issue https://github.com/NVIDIA/MinkowskiEngine/issues/306
    When fixed, delete this function and replace ME_sparse_to_dense(x) with x.dense() everywhere in code.
    """
    if min_coordinate is not None:
        assert isinstance(min_coordinate, (torch.IntTensor, torch.cuda.IntTensor))
        assert min_coordinate.numel() == x._D
    if shape is not None:
        assert isinstance(shape, torch.Size)
        assert len(shape) == x._D + 2  # batch and channel
        if shape[1] != x._F.size(1):
            shape = torch.Size([shape[0], x._F.size(1), *[s for s in shape[2:]]])

    # Use int tensor for all operations
    tensor_stride = torch.tensor(x.tensor_stride, device=x.device)

    # New coordinates
    batch_indices = x.C[:, 0]

    # shift by min coordinate
    if min_coordinate is None:
        min_coordinate, _ = x.C.min(0, keepdim=True)
        min_coordinate = min_coordinate[:, 1:]
        coords = x.C[:, 1:] - min_coordinate
    elif isinstance(min_coordinate, int) and min_coordinate == 0:
        coords = x.C[:, 1:]
    else:
        if min_coordinate.ndim == 1:
            min_coordinate = min_coordinate.unsqueeze(0)
        coords = x.C[:, 1:] - min_coordinate

    assert (min_coordinate % tensor_stride).sum() == 0, "The minimum coordinates must be divisible by the tensor stride."


    if coords.ndim == 1:
        coords = coords.unsqueeze(1)

    # return the contracted tensor
    if contract_stride:
        coords = coords // tensor_stride

    # clamp if > shape's max coordinate
    max_coordinate = torch.tensor(shape[2:], device=coords.device, dtype=coords.dtype)
    mask = torch.where((coords < max_coordinate).all(axis=1))[0]
    coords = coords[mask]
    feats = x.F[mask]
    batch_indices = batch_indices[mask]

    nchannels = x.F.size(1)
    if shape is None:
        size = coords.max(0)[0] + 1
        shape = torch.Size([batch_indices.max() + 1, nchannels, *size.cpu().numpy()])

    dense_F = torch.zeros(shape, dtype=x.F.dtype, device=x.F.device)


    tcoords = coords.t().long()
    batch_indices = batch_indices.long()
    exec(
        "dense_F[batch_indices, :, "
        + ", ".join([f"tcoords[{i}]" for i in range(len(tcoords))])
        + "] = feats"
    )


    tensor_stride = torch.IntTensor(x.tensor_stride)
    return dense_F, min_coordinate, tensor_stride