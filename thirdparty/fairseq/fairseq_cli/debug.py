# encoding: utf-8
"""



@desc: 

"""

import numpy as np
#
# file = "/data/nfsdata2/nlp_application/datasets/nmt/iwslt/iwslt14/iwslt14.pretrained_wmt19.de-en/iwslt14.tokenized.de-en/de-en-bin/test-features/all.mmap.encoder.bak"
#
# x = np.memmap(
#     file,
#     dtype=np.float32, mode='r',
#     shape=(162040, 1024)
# )
#
# file2 = "/data/nfsdata2/nlp_application/datasets/nmt/iwslt/iwslt14/iwslt14.pretrained_wmt19.de-en/iwslt14.tokenized.de-en/de-en-bin/test-features/all.mmap.encoder"
#
# y = np.memmap(
#     file2,
#     dtype=np.float32, mode='r',
#     shape=(162040, 1024)
# )
#
# for i in range(x.shape[0]):
#     if not np.allclose(x[i], y[i], rtol=1.e-3, atol=1.e-4):
#         print(i)
#         print(x[i][: 10])
#         print(y[i][: 10])


file = "/data/yuxian/debug.mmap"

n = 10
d = 5
x = np.memmap(
    file,
    dtype=np.float32, mode='r+',
    shape=(n+10, d)
)
print(x)

