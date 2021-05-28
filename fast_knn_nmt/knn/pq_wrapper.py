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
        self.register_buffer("centroids_torch", torch.from_numpy(self.centroids))
        self.register_buffer("norm2_centroids_torch", torch.from_numpy(self.norm2_centroids))

    def encode(self, x):
        """

        Args:
            x: [n, M]

        Returns:

        """
        if self.pre_torch:
            A, b = self.A, self.b
            x = x @ A.t()
            if b.numel() > 0:
                x += b

        n, d = x.shape
        cen = self.centroids_torch
        M, ksub, dsub = cen.shape
        codes = torch.empty((n, M), dtype=torch.uint8, device=x.device)
        # maybe possible to vectorize this loop...
        for m in range(M):
            # compute all per-centroid distances, ignoring the ||x||^2 term
            xslice = x[:, m * dsub:(m + 1) * dsub]  # []
            dis = self.norm2_centroids_torch[m] - 2 * xslice @ cen[m].t()
            codes[:, m] = dis.argmin(dim=1)
        return codes

    def decode(self, codes):
        n, MM = codes.shape
        cen = self.centroids_torch
        M, ksub, dsub = cen.shape
        assert MM == M
        x = torch.empty((n, M * dsub), dtype=torch.float32, device=codes.device)
        for m in range(M):  # todo: vectorize
            xslice = cen[m, codes[:, m].long()]
            x[:, m * dsub:(m + 1) * dsub] = xslice

        if self.pre is not None:
            A, b = self.A, self.b
            if b.numel() > 0:
                x -= b
            x = x @ A
        return x


if __name__ == '__main__':
    # codec = faiss.index_factory(128, "OPQ32_128,PQ32")
    codec = faiss.index_factory(128, "PCAR64,PQ32")
    # train the codec
    xt = faiss.rand((10000, 128))
    codec.train(xt)
    # convetert to a pure numpy codec
    ncodec = NumpyPQCodec(codec)
    # test encoding
    xb = faiss.rand((100, 128), 1234)
    (ncodec.encode(xb) == codec.sa_encode(xb)).all()
    # test decoding
    codes = faiss.randint((100, 32), vmax=256, seed=12345).astype('uint8')
    np.allclose(codec.sa_decode(codes), ncodec.decode(codes))

    # test encode
    dev = torch.device('cuda:0')
    tcodec = TorchPQCodec(codec).to(dev)

    xb = torch.rand(1000, 128).to(dev)
    (tcodec.encode(xb).cpu().numpy() == codec.sa_encode(xb.cpu().numpy())).all()
    # test decode
    codes = torch.randint(256, size=(1000, 32), dtype=torch.uint8).to(dev)
    np.allclose(
        tcodec.decode(codes).cpu().numpy(),
        codec.sa_decode(codes.cpu().numpy()),
        atol=1e-6
    )
