# encoding: utf-8
"""
@desc: https://gist.github.com/mdouze/94bd7a56d912a06ac4719c50fa5b01ac

"""

import faiss
import numpy as np
import torch


class NumpyPQCodec:

    def __init__(self, index):

        assert index.is_trained

        # handle the pretransform
        if isinstance(index, faiss.IndexPreTransform):
            vt = faiss.downcast_VectorTransform(index.chain.at(0))
            assert isinstance(vt, faiss.LinearTransform)
            b = faiss.vector_to_array(vt.b)
            A = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
            self.pre = (A, b)
            index = faiss.downcast_index(index.index)
        else:
            self.pre = None

        # extract the PQ centroids
        assert isinstance(index, faiss.IndexPQ)
        pq = index.pq
        cen = faiss.vector_to_array(pq.centroids)
        cen = cen.reshape(pq.M, pq.ksub, pq.dsub)
        assert pq.nbits == 8
        self.centroids = cen
        self.norm2_centroids = (cen ** 2).sum(axis=2)

    def encode(self, x):
        if self.pre is not None:
            A, b = self.pre
            x = x @ A.T
            if b.size > 0:
                x += b

        n, d = x.shape
        cen = self.centroids
        M, ksub, dsub = cen.shape
        codes = np.empty((n, M), dtype='uint8')
        # maybe possible to vectorize this loop...
        for m in range(M):
            # compute all per-centroid distances, ignoring the ||x||^2 term
            xslice = x[:, m * dsub:(m + 1) * dsub]
            dis = self.norm2_centroids[m] - 2 * xslice @ cen[m].T
            codes[:, m] = dis.argmin(axis=1)
        return codes

    def decode(self, codes):
        n, MM = codes.shape
        cen = self.centroids
        M, ksub, dsub = cen.shape
        assert MM == M
        x = np.empty((n, M * dsub), dtype='float32')
        for m in range(M):
            xslice = cen[m, codes[:, m]]
            x[:, m * dsub:(m + 1) * dsub] = xslice
        if self.pre is not None:
            A, b = self.pre
            if b.size > 0:
                x -= b
            x = x @ A
        return x


class TorchPQCodec(NumpyPQCodec, torch.nn.Module):

    def __init__(self, index):
        NumpyPQCodec.__init__(self, index)
        torch.nn.Module.__init__(self)
        # just move everything to torch on the given device
        if self.pre:
            A, b = self.pre
            self.pre_torch = True
            self.register_buffer("A", torch.from_numpy(A))
            self.register_buffer("b", torch.from_numpy(b))
        else:
            self.pre_torch = False
        self.register_buffer("centroids_torch", torch.from_numpy(self.centroids))  # [M, ksub, dsub]
        self.register_buffer("norm2_centroids_torch", torch.from_numpy(self.norm2_centroids))  # [M, ksub]

    def encode(self, x):
        """

        Args:
            x: [n, D], where D = M * dsub. M is the number of subspaces, dsub is the dimension of each subspace

        Returns:

        """
        if self.pre_torch:
            A, b = self.A, self.b
            x = x @ A.t()
            if b.numel() > 0:
                x += b

        n, d = x.shape
        cen = self.centroids_torch  # [M, ksub, dsub]
        M, ksub, dsub = cen.shape

        # for loop version
        # codes = torch.empty((n, M), dtype=torch.uint8, device=x.device)
        # # maybe possible to vectorize this loop...
        # for m in range(M):
        #     # compute all per-centroid distances, ignoring thecen.shape ||x||^2 term
        #     xslice = x[:, m * dsub:(m + 1) * dsub]  # [n, dsub]
        #     dis = self.norm2_centroids_torch[m] - 2 * xslice @ cen[m].t()
        #     codes[:, m] = dis.argmin(dim=1)

        # parallel version
        x = x.view(n, M, dsub).unsqueeze(-2)  # [n, M, 1, dsub]
        norm = self.norm2_centroids_torch.unsqueeze(0)  # [1, M, ksub]
        cen = cen.transpose(1, 2).unsqueeze(0)  # [1, M, dsub, ksub]
        dot_product = torch.matmul(x, cen).squeeze(-2)  # [n, M, 1, ksub]
        dis = norm - 2 * dot_product  # [n, M, ksub]
        codes = dis.argmin(dim=2).to(torch.uint8)

        return codes

    def decode(self, codes):
        """

        Args:
            codes: [n, M], where M is the number of subspaces
        Returns:
            feature: [n, D], where D = M * dsub, and dsub is the dimension of subspace

        """
        n, MM = codes.shape
        cen = self.centroids_torch   # [M, ksub, dsub]
        M, ksub, dsub = cen.shape
        assert MM == M

        # for loop version
        # x = torch.empty((n, M * dsub), dtype=torch.float32, device=codes.device)
        # for m in range(M):
        #     xslice = cen[m, codes[:, m].long()]
        #     x[:, m * dsub:(m + 1) * dsub] = xslice

        # parallel version
        # x[n, m, j] = cen[m][codes[n][m]][j]
        x = torch.gather(
            cen.unsqueeze(0).expand(n, -1, -1, -1),  # [n, M, ksub, dsub]
            dim=2,
            index=codes.long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, dsub)  # [n, M, 1, dsub]
        )   # [n, M, 1, dsub]
        x = x.view(n, -1)

        if self.pre is not None:
            A, b = self.A, self.b
            if b.numel() > 0:
                x -= b
            x = x @ A
        return x


if __name__ == '__main__':
    codec = faiss.read_index("//data/yuxian/datasets/multi_domain_paper/it/bpe/de-en-bin/quantizer-decoder.new")
    # test encode
    dev = torch.device('cuda:0')
    tcodec = TorchPQCodec(codec).to(dev)

    xb = torch.rand(1000, 1024).to(dev)
    (tcodec.encode(xb).cpu().numpy() == codec.sa_encode(xb.cpu().numpy())).all()
    # test decode
    codes = torch.randint(256, size=(1000, 128), dtype=torch.uint8).to(dev)
    np.allclose(
        tcodec.decode(codes).cpu().numpy(),
        codec.sa_decode(codes.cpu().numpy()),
        atol=1e-6
    )

    from tqdm import tqdm

    for _ in tqdm(range(100)):
        tcodec.encode(xb)

    for _ in tqdm(range(100)):
        tcodec.decode(codes)
