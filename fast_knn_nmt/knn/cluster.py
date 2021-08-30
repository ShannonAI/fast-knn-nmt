# encoding: utf-8
"""



@desc: k-means in source / target side

"""

import os
import argparse
import json
import numpy as np
from tqdm import tqdm
import multiprocessing
import faiss

from fast_knn_nmt.data.utils import warmup_mmap_file

def get_info(data_store):
    new_line = ""
    with open(os.path.join(data_store, "info.json"), "r") as f:
        for line in f:
            new_line += line
        f.close()
    return json.loads(new_line)

def build_cluster(data_store, cluster_size, n_iter=100, use_gpu=False, queue=None):
    info = get_info(data_store)
    key_file = os.path.join(data_store, "keys.npy")

    warmup_mmap_file(key_file, verbose=False, use_log=False)

    keys = np.memmap(key_file, 
                     dtype=np.float16 if info['dstore_fp16'] else np.float32,
                     mode="r",
                     shape=(info['dstore_size'], info['hidden_size']))
    train_in_memory = np.zeros((info['dstore_size'], info['hidden_size']), dtype=np.float32)
    train_in_memory[:] = keys[:]

    n_cluster = info['dstore_size'] // cluster_size
    if (n_cluster * cluster_size != info['dstore_size']):
        n_cluster += 1
    '''
    k_means = faiss.Kmeans(d=info['hidden_size'], k=n_cluster, niter=n_iter, verbose=True, gpu=use_gpu)
    k_means.train(train_in_memory)
    '''

    clus = faiss.Clustering(info['hidden_size'], n_cluster)
    clus.seed = np.random.randint(1234)
    clus.niter = n_iter
    clus.max_points_per_centroid = 10000000
    index = faiss.IndexFlatL2(info['hidden_size'])
    if use_gpu:
        index = faiss.index_cpu_to_all_gpus(index, ngpu=faiss.get_num_gpus())
    clus.train(train_in_memory, index)
    

    cluster_file = os.path.join(data_store, "cluster_center.npy")
    cluster_mmap = np.memmap(cluster_file, 
                     dtype=np.float16 if info['dstore_fp16'] else np.float32,
                     mode="w+",
                     shape=(n_cluster, info['hidden_size']))

    #cluster_mmap[:] = k_means.centroids[:]
    cluster_mmap[:] = faiss.vector_to_array(clus.centroids).reshape(n_cluster, info['hidden_size'])[:]
    
    center_has = [[] for i in range(n_cluster)]
    
    #D, labels = k_means.index.search(train_in_memory, 1)
    D, labels = index.search(train_in_memory, 1)
    for i in range(info['dstore_size']):
        center_has[int(labels[i][0])].append((i, float(D[i][0])))

    def takeSecond(elem):
        return elem[1]

    '''
    for i in range(n_cluster):
        if (len(center_has[i]) > cluster_size):
            center_has[i].sort(key=takeSecond)
            center_has[i] = center_has[i][:cluster_size]
    '''

    cluster_key_file = os.path.join(data_store, "cluster_key_offset.json")
    json.dump(center_has, open(cluster_key_file, "w"),
                  sort_keys=True, indent=4, ensure_ascii=False)
        
    cluster_info = {
        "n_cluster": n_cluster,
        "hidden_size": info['hidden_size'],
        "val_size": info['val_size'],
        "cluster_size": [len(point_list) for point_list in center_has],
        "dstore_fp16": info['dstore_fp16'] 
    }
    json.dump(cluster_info, open(os.path.join(data_store, "cluster_info.json"), "w"),
                  sort_keys=True, indent=4, ensure_ascii=False)
    if queue is not None:
        queue.put(1)

def main():
    parser = argparse.ArgumentParser(description='building cluster')
    parser.add_argument("--dstore-dir", type=str, help="paths to data store. if provided multiple,"
                                                                      "use ',' as separator")
    parser.add_argument('--cluster-size', type=int, default=512, help='cluster size')
    parser.add_argument('--num-workers', type=int, default=0, help="thread_num")
    parser.add_argument("--use_gpu", default=False, action="store_true",
                        help="to run k-means on gpu")
    
    args = parser.parse_args()

    file_list = []
    files = os.listdir(args.dstore_dir)  
    for f in files:    
        file = os.path.join(args.dstore_dir, f)    
        if os.path.isdir(file):
            file_list.append(file)
        
    if (args.num_workers <= 1):
        with tqdm(total=len(file_list)) as pbar:
            for file in file_list:
                print(file)
                build_cluster(data_store=file, cluster_size=args.cluster_size, use_gpu=args.use_gpu)
                pbar.update(1)
    else:
        pool = multiprocessing.Pool(args.num_workers)
        queue = multiprocessing.Manager().Queue()

        for file in file_list:
            pool.apply_async(build_cluster, args=(file, args.cluster_size, 100, args.use_gpu, queue,))
        
        pool.close()

        pbar = tqdm(total=len(file_list))
        cnt = 0
        while (True):
            cnt += queue.get()
            pbar.update(1)
            if (cnt >= len(file_list)):
                break
        pbar.close()
        pool.join()



if __name__ == '__main__':
    main()

'''
# 1. build datastores for each token
python fast_knn_nmt/knn/buid_ds.py \
--data_dir $DATA_DIR \
--prefix $PREFIX \
--lang $SRC_LANG \
--mode "train" --workers 16 --offset_chunk 1000000  --use_memory


# 2. k-means
python ./fast_knn_nmt/knn/cluster.py \
--dstore-dir $DATA_DIR/train_de_data_stores \
--cluster-size 512 \
--num-workers 12

# 3. build new faiss indexes only for train
DS_DIRS=$DATA_DIR/train_${SRC_LANG}_data_stores
metric="cosine"
index="auto"
python fast_knn_nmt/knn/run_index_build.py \
  --dstore-dir $DS_DIRS  --workers 0 \
  --index-type $index --chunk-size 200000 \
  --subdirs --metric $metric  --overwrite \
  --use-cluster

# 4. find knn neighbors for each token
metric="cosine"
k=512
for mode in "test"; do
python fast_knn_nmt/knn/find_knn_neighbors.py \
--data_dir $DATA_DIR \
--prefix $PREFIX \
--lang $SRC_LANG --use_memory --offset_chunk 1000000 \
--mode $mode --workers 0 --k $k --metric $metric --nprobe 32 --use-gpu \
--use-cluster
done


# quantize decoder feature
index="PQ128"
#index="OPQ128_512,,PQ128"
python fast_knn_nmt/knn/quantize_features2.py \
--data-dir $DATA_DIR  \
--prefix $PREFIX \
--lang $TGT_LANG \
--subset "train" \
--chunk-size 1000000 \
--index $index \
--compute-error

'''