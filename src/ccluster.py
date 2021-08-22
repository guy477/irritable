import ccluster
import numpy as np
import concurrent.futures
from sklearn.cluster import MiniBatchKMeans

# 

def nump2(n, k):
    a = np.ones((k, n-k+1), dtype=np.int16)
    a[0] = np.arange(n-k+1)
    for j in range(1, k):
        reps = (n-k+j) - a[j-1]
        a = np.repeat(a, reps, axis=1)
        ind = np.add.accumulate(reps)
        a[j, ind[:-1]] = 1-reps[1:]
        a[j, 0] = j
        a[j] = np.add.accumulate(a[j])
    return a.T

k = 5
n = 52
# n = 15

dat_y = nump2(n, k)

x = nump2(n, 2)


# y = np.memmap('/media/poker_raw/river.npy', mode = 'w+', dtype = np.uint8, shape = (dat_y.shape[0],5))
# y[:] = dat_y[:]
# y.flush()

crnt = (16)
chunksize = len(x) // crnt
save = list(range(crnt))

# with concurrent.futures.ProcessPoolExecutor() as executor:
#     # change crnt to appropriate number of workers for your system
#     futures = []
#     for i in range(crnt):
        
#         strt = i * chunksize
#         stp = ((i + 1) * chunksize) if i != (crnt - 1) else len(x)
        
#         futures.append(executor.submit(ccluster.do_calc, x[strt:stp], dat_y, i))
#     concurrent.futures.wait(futures)

#     output = [f.result() for f in futures]


# print(ccluster.do_calc(x, dat_y, 69420))

ranks = ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')
suits = ('c', 'd', 'h', 's')

for i in range(crnt):
    strt = i * chunksize
    stp = ((i + 1) * chunksize) if i != (crnt - 1) else len(x)
    
    # pylint: disable=unexpected-keyword-arg
    save[i] = (i, np.load("results/zfin-" + str(i) + ".npy", mmap_mode='r'))


for i in save:
    print(i[1])

#print(tmp[:, 7:].shape)
# seeding = ccluster.kmc2(tmp[:, 7:], 2000)
#print(seeding)
#model = MiniBatchKMeans(2000, init=seeding).fit(tmp[:, 7:])
#print(model.cluster_centers_)



# for i in save:
#     mat = np.load('results/zfin-'+str(i)+'.npy')
#     c = 0
#     ind = 0
#     while c < 100:
#         if (mat[ind][0] != mat[ind][1]):
#             for j in range(10):
#                 print(mat[ind][j], end = ' ')
#                 c += 1
#             print()
#         ind += 1
#     c = 0
#     ind = 1
#     while c < 100:
#         if (mat[-ind][0] != mat[-ind][1]):
#             for j in range(10):
#                 print(mat[-ind][j], end = ' ')
#                 c += 1
#             print()
#         ind += 1
#     print()
#     print()
#
# for i in range(52):
#     print(str(i) + ' - ' + ranks[i%13]+suits[i//13])

# print(ccluster.do_calc(1, 45, 6))