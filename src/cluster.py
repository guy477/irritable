import numpy as np
import concurrent.futures
import holdem_calc
import threading
import itertools
import timeit
import struct
from eval7 import Card, evaluate


class cluster:


    def __init__(self):
        self.rank = {0 : '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: 'T', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'}
        self.suit = {0: 's', 1: 'h', 2: 'd', 3: 'c'}
        self.run()

        

    def run(self):
        
        threads = 8

        k = 5
        n = 52
        # n = 15

        self.dat_y = self.nump2(n, k)
        self.dat_yy = self.nump2(n, k-1)
        self.dat_yyy = self.nump2(n, k-2)

        


        print(self.dat_y.shape)

        x = self.nump2(n, 2)

        print(x)
        print(len(x))
        print(x.shape)

        
        # y = np.memmap('/media/poker_raw/river.npy', mode = 'r+', dtype = np.uint8, shape = (2598960,5))
        y = np.memmap('/media/poker_raw/river.npy', mode = 'r+', dtype = np.uint8, shape = (self.dat_y.shape[0],5))        
        yy = np.memmap('/media/poker_raw/turn.npy', mode = 'r+', dtype = np.uint8, shape = (self.dat_yy.shape[0],4))
        yyy = np.memmap('/media/poker_raw/flop.npy', mode = 'r+', dtype = np.uint8, shape = (self.dat_yyy.shape[0],3))
        
        y[:] = self.dat_y[:]
        yy[:] = self.dat_yy[:]
        yyy[:] = self.dat_yyy[:]   

        y.flush()
        yy.flush()
        yyy.flush()

        print(y)
        print(len(y))
        print(y.shape)        

        # # z = np.memmap('/media/poker_raw/cartesianRiver.npy', mode = 'w+', dtype = np.uint8, shape = (y.shape[0] * x.shape[0], y.shape[1] + x.shape[1]))
        
        
        
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     # change crnt to appropriate number of workers for your system
        #     crnt = (threads - 1)
        #     futures = []
        #     chunksize = len(x) // crnt
            
        #     for i in range(crnt):
                
        #         strt = i * chunksize
        #         stp = ((i + 1) * chunksize) if i != (crnt - 1) else len(x)
        #         # print(x[strt:stp])
        #         futures.append(executor.submit(self.par_combine_river ,x,strt,stp))
        #     concurrent.futures.wait(futures)

        #     output = [f.result() for f in futures]

        #     print(output)
        #     print(sum(output))

        z = np.memmap('/media/poker_raw/cartesianRiver.npy', mode = 'r', dtype = np.uint8, shape = (y.shape[0] * x.shape[0], y.shape[1] + x.shape[1]))
        
        self.numbad = 0
        # 480200 bad datapoints per starting hand
        for i in range(0, self.dat_y.shape[0]):
            if(self.contains_duplicates(z[i])):
                self.numbad += 1
        print(self.numbad)


        # xyz = np.memmap('/media/poker_raw/PMAP_River.npy', mode = 'w+', dtype = np.uint8, shape = ((y.shape[0] - self.numbad) * x.shape[0], y.shape[1] + x.shape[1] + 8))

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     # change crnt to appropriate number of workers for your system
        #     crnt = (threads - 1)
        #     futures = []
        #     chunksize = len(x) // crnt
            
        #     for i in range(crnt):
                
        #         strt = i * chunksize
        #         stp = ((i + 1) * chunksize) if i != (crnt - 1) else len(x)
        #         # print(x[strt:stp])
        #         futures.append(executor.submit(self.par_combine_filter_river ,x,strt,stp))
        #     concurrent.futures.wait(futures)

        #     output = [f.result() for f in futures]

        #     print(output)
        #     # print(sum(output))

        self.mp_hand = self.map_indices_by_hand(x)
        self.mp_table_river = self.map_indices(self.dat_y)
        self.mp_table_turn = self.map_indices(self.dat_yy)
        self.mp_table_flop = self.map_indices(self.dat_yyy)
        self.combine_filter_river(x)

        xyz = np.memmap('/media/poker_raw/PMAP_River.npy', mode = 'r', dtype = np.uint8, shape = ((y.shape[0] - self.numbad) * x.shape[0], y.shape[1] + x.shape[1] + 8))
        print(xyz)
        

        
        # # # zz = np.memmap('/media/poker_raw/cartesianTurn.npy', mode = 'w+', dtype = np.uint8, shape = (yy.shape[0] * x.shape[0], yy.shape[1] + x.shape[1]))
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     # change crnt to appropriate number of workers for your system
        #     crnt = (threads - 1)
        #     futures = []
        #     chunksize = len(x) // crnt
            
        #     for i in range(crnt):
                
        #         strt = i * chunksize
        #         stp = ((i + 1) * chunksize) if i != (crnt - 1) else len(x)
        #         # print(x[strt:stp])
        #         futures.append(executor.submit(self.par_combine_turn ,x,strt,stp))
        #     concurrent.futures.wait(futures)

        #     output = [f.result() for f in futures]

        #     print(output)
        #     print(sum(output))

        # zz = np.memmap('/media/poker_raw/cartesianTurn.npy', mode = 'r', dtype = np.uint8, shape = (yy.shape[0] * x.shape[0], yy.shape[1] + x.shape[1]))
        # # self.numbad = 480200

        # self.numbad = 0
        # for i in range(0, self.dat_yy.shape[0]):
        #     if(self.contains_duplicates(zz[i])):
        #         self.numbad += 1
        # print(self.numbad)


        
        # xyyz = np.memmap('/media/poker_raw/PMAP_turn.npy', mode = 'w+', dtype = np.uint8, shape = ((yy.shape[0] - self.numbad) * x.shape[0], yy.shape[1] + x.shape[1] + 8))
        
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     # change crnt to appropriate number of workers for your system
        #     crnt = (threads - 1)
        #     futures = []
        #     chunksize = len(x) // crnt
            
        #     for i in range(crnt):
                
        #         strt = i * chunksize
        #         stp = ((i + 1) * chunksize) if i != (crnt - 1) else len(x)
        #         # print(x[strt:stp])
        #         futures.append(executor.submit(self.par_combine_filter_turn ,x,strt,stp))
        #     concurrent.futures.wait(futures)

        #     output = [f.result() for f in futures]

        #     print(output)
        #     # print(sum(output))


        # # # zzz = np.memmap('/media/poker_raw/cartesianFlop.npy', mode = 'w+', dtype = np.uint8, shape = (yyy.shape[0] * x.shape[0], yyy.shape[1] + x.shape[1]))
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     # change crnt to appropriate number of workers for your system
        #     crnt = (threads - 1)
        #     futures = []
        #     chunksize = len(x) // crnt
            
        #     for i in range(crnt):
                
        #         strt = i * chunksize
        #         stp = ((i + 1) * chunksize) if i != (crnt - 1) else len(x)
        #         # print(x[strt:stp])
        #         futures.append(executor.submit(self.par_combine_flop ,x,strt,stp))
        #     concurrent.futures.wait(futures)

        #     output = [f.result() for f in futures]

        #     print(output)
        #     print(sum(output))

        # zzz = np.memmap('/media/poker_raw/cartesianFlop.npy', mode = 'r', dtype = np.uint8, shape = (yyy.shape[0] * x.shape[0], yyy.shape[1] + x.shape[1]))
        # # self.numbad = 480200

        # self.numbad = 0
        # for i in range(0, self.dat_yyy.shape[0]):
        #     if(self.contains_duplicates(zzz[i])):
        #         self.numbad += 1
        # print(self.numbad)


        
        # xyyyz = np.memmap('/media/poker_raw/PMAP_flop.npy', mode = 'w+', dtype = np.uint8, shape = ((yyy.shape[0] - self.numbad) * x.shape[0], yyy.shape[1] + x.shape[1] + 8))
        

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     # change crnt to appropriate number of workers for your system
        #     crnt = (threads - 1)
        #     futures = []
        #     chunksize = len(x) // crnt
            
        #     for i in range(crnt):
                
        #         strt = i * chunksize
        #         stp = ((i + 1) * chunksize) if i != (crnt - 1) else len(x)
        #         # print(x[strt:stp])
        #         futures.append(executor.submit(self.par_combine_filter_flop ,x,strt,stp))
        #     concurrent.futures.wait(futures)

        #     output = [f.result() for f in futures]

        #     print(output)
        #     # print(sum(output))


        
        # self.mp_hand = self.map_indices_by_hand(x)
        # self.mp_table_river = self.map_indices(self.dat_y)
        # self.mp_table_turn = self.map_indices(self.dat_yy)
        # self.mp_table_flop = self.map_indices(self.dat_yyy)
        

        # print(z)
        # print(zz)
        # print(zzz)

        # print(xyz)
        # print(xyyz)
        # print(xyyyz)

    
    def get_ehs_fast(self, combo):
        x = list(range(52))
        j = combo.tolist()
        for i in j:
            x.remove(i)

        hero = evaluate([Card(self.get_card(s)) for s in j])
        vil = -1
        wlt = [0, 0, 0]
        for i in range(0, len(x)-1):
            for k in range(i+1, len(x)):
                
                vil = evaluate([Card(self.get_card(s)) for s in j[2:] + [x[i], x[k]]])
                if(hero > vil):
                    wlt[1] += 1
                elif(vil > hero):
                    wlt[-1] += 1
                else:
                    wlt[0] += 1
        
        s = sum(wlt)
        wlt[0] /= s
        wlt[1] /= s
        wlt[2] /= s
        return wlt

    def get_ehs(self, combo):
        x = list(range(52))
        k = combo
        for i in k:
            x.remove(i)
        wlt = [0, 0, 0]
        for i in range(0, len(x)-1):
            for j in range(i+1, len(x)):
                # print([get_card(x) for x in board])
                # print([get_card(x) for x in hand + [x[i], x[j]]])
                w = holdem_calc.calculate(board = [self.get_card(x) for x in k[2:]], exact = True, hole_cards = [self.get_card(x) for x in k[:2] + [x[i], x[j]]], verbose = False, num = 10, input_file = None)
                wlt[0] += w[0]
                wlt[1] += w[1]
                wlt[2] += w[2]
                
        s = sum(wlt)
        wlt[0] /= s
        wlt[1] /= s
        wlt[2] /= s
        return wlt

    def nump2(self, n, k):
        a = np.ones((k, n-k+1), dtype=np.uint8)
        a[0] = np.arange(n-k+1)
        for j in range(1, k):
            reps = (n-k+j) - a[j-1]
            a = np.repeat(a, reps, axis=1)
            ind = np.add.accumulate(reps)
            a[j, ind[:-1]] = 1-reps[1:]
            a[j, 0] = j
            a[j] = np.add.accumulate(a[j])
        return a.T

    def contains_duplicates(self, X):
        seen = set()
        seen_add = seen.add
        for x in X:
            if (x in seen or seen_add(x)):
                return True
        return False


    def par_combine_river(self, x, start, stop):
        xx = x[start:stop]
        y = np.memmap('/media/poker_raw/river.npy', mode = 'r', dtype = np.uint8, shape = (self.dat_y.shape[0],5))
        z = np.memmap('/media/poker_raw/cartesianRiver.npy', mode = 'r+', dtype = np.uint8, shape = (y.shape[0] * xx.shape[0], xx.shape[1] + y.shape[1]), offset = (start) * y.shape[0] * 7)
        idx = 0
        self.numbad = 0
        for i in xx.tolist():
            for j in y.tolist():
                k = np.concatenate((i, j))
                z[idx] = k
                idx += 1
            print(i)
            z.flush
        
        print('done - ' + str(start) + ' - ' + str(len(x)))

        z.flush()
        return self.numbad

    def par_combine_turn(self, x, start, stop):
        xx = x[start:stop]
        y = np.memmap('/media/poker_raw/turn.npy', mode = 'r', dtype = np.uint8, shape = (self.dat_yy.shape[0],4))
        z = np.memmap('/media/poker_raw/cartesianTurn.npy', mode = 'r+', dtype = np.uint8, shape = (y.shape[0] * xx.shape[0], xx.shape[1] + y.shape[1]), offset = (start) * y.shape[0] * 6)
        idx = 0
        self.numbad = 0
        for i in xx.tolist():
            for j in y.tolist():
                k = np.concatenate((i, j))
                z[idx] = k
                idx += 1
            print(i)
            z.flush
        
        print('done - ' + str(start) + ' - ' + str(len(x)))

        z.flush()
        return self.numbad

    def par_combine_flop(self, x, start, stop):
        xx = x[start:stop]
        y = np.memmap('/media/poker_raw/flop.npy', mode = 'r', dtype = np.uint8, shape = (self.dat_yyy.shape[0],3))
        z = np.memmap('/media/poker_raw/cartesianFlop.npy', mode = 'r+', dtype = np.uint8, shape = (y.shape[0] * xx.shape[0], xx.shape[1] + y.shape[1]), offset = (start) * y.shape[0] * 5)
        idx = 0
        self.numbad = 0
        for i in xx.tolist():
            for j in y.tolist():
                k = np.concatenate((i, j))
                z[idx] = k
                idx += 1
            print(i)
            z.flush
        
        print('done - ' + str(start) + ' - ' + str(len(x)))

        z.flush()
        return self.numbad    

    def get_card(self, x):
        
        return (self.rank[x%13] + self.suit[x//13])

    # using the mappings and updating everything at once, receive exponential speed up
    def combine_filter_river(self, x):
        y = np.memmap('/media/poker_raw/river.npy', mode = 'r', dtype = np.uint8, shape = (self.dat_y.shape[0],5))
        z = np.memmap('/media/poker_raw/cartesianRiver.npy', mode = 'r', dtype = np.uint8, shape = (y.shape[0] * x.shape[0], x.shape[1] + y.shape[1]))
        xyz = np.memmap('/media/poker_raw/PMAP_River.npy', mode = 'r+', dtype = np.uint8, shape = ((self.dat_y.shape[0] - self.numbad) * x.shape[0], self.dat_y.shape[1] + x.shape[1] + 8))
        idx = 0
        bite = []
        print((y.shape[0] - self.numbad))
        for i in z:
            if(not self.contains_duplicates(i)):
                bite = self.get_ehs_fast(i)

                xyz[idx][:7] = i
                xyz[idx][7:11] = [c for c in struct.pack('!f', bite[0])]
                xyz[idx][11:] = [c for c in struct.pack('!f', bite[1])]
                                
                # print(struct.unpack('!f', xyz[idx][-4:]))
                # print(str([self.get_card(x) for x in xyz[idx][:7]]) + ' - ' + str(struct.unpack('!f', xyz[idx][-8:-4])[0]) + str(struct.unpack('!f', xyz[idx][-4:])[0]))
                # print(struct.unpack('!f', xyz[idx][-8:-4])[0])
                # print(struct.unpack('!f', xyz[idx][-4:])[0])
                
                idx += 1
                # print(idx)
                if(((idx + 1) % (y.shape[0] - self.numbad)) == 0):
                    print(xyz[idx])
                    xyz.flush()

    def par_combine_filter_river(self, x, start, stop):
        xx = x[start:stop]
        y = np.memmap('/media/poker_raw/river.npy', mode = 'r', dtype = np.uint8, shape = (self.dat_y.shape[0],5))
        z = np.memmap('/media/poker_raw/cartesianRiver.npy', mode = 'r', dtype = np.uint8, shape = (y.shape[0] * xx.shape[0], xx.shape[1] + y.shape[1]), offset = (start) * y.shape[0] * 7)
        xyz = np.memmap('/media/poker_raw/PMAP_River.npy', mode = 'r+', dtype = np.uint8, shape = ((self.dat_y.shape[0] - self.numbad) * xx.shape[0], self.dat_y.shape[1] + xx.shape[1] + 8), offset = (start) * (y.shape[0]-self.numbad) * (self.dat_y.shape[1] + xx.shape[1] + 8))
        idx = 0
        bite = []
        print((y.shape[0] - self.numbad))
        for i in z:
            if(not self.contains_duplicates(i)):
                bite = self.get_ehs_fast(i)

                xyz[idx][:7] = i
                xyz[idx][7:11] = [c for c in struct.pack('!f', bite[0])]
                xyz[idx][11:] = [c for c in struct.pack('!f', bite[1])]
                                
                # print(struct.unpack('!f', xyz[idx][-4:]))
                # print(str([self.get_card(x) for x in xyz[idx][:7]]) + ' - ' + str(struct.unpack('!f', xyz[idx][-8:-4])[0]) + str(struct.unpack('!f', xyz[idx][-4:])[0]))
                # print(struct.unpack('!f', xyz[idx][-8:-4])[0])
                # print(struct.unpack('!f', xyz[idx][-4:])[0])
                
                idx += 1
                # print(idx)
                if(((idx + 1) % (y.shape[0] - self.numbad)) == 0):
                    print(xyz[idx])
                    xyz.flush()
            
        
        xyz.flush()
        


    def par_combine_filter_turn(self, x, start, stop):
        xx = x[start:stop]
        y = np.memmap('/media/poker_raw/turn.npy', mode = 'r', dtype = np.uint8, shape = (self.dat_yy.shape[0],4))
        z = np.memmap('/media/poker_raw/cartesianTurn.npy', mode = 'r', dtype = np.uint8, shape = (y.shape[0] * xx.shape[0], xx.shape[1] + y.shape[1]), offset = (start) * y.shape[0] * 6)
        xyz = np.memmap('/media/poker_raw/PMAP_Turn.npy', mode = 'r+', dtype = np.uint8, shape = ((self.dat_yy.shape[0] - self.numbad) * xx.shape[0], self.dat_yy.shape[1] + xx.shape[1] + 8))
        idx = 0
        j = 0
        for i in z.tolist():
            if(self.contains_duplicates(i)):
                xyz[idx] = i
                idx += 1
                if((j + 1) % y.shape[0] == 0):
                    xyz.flush()
        
        xyz.flush()

    def par_combine_filter_flop(self, x, start, stop):
        xx = x[start:stop]
        y = np.memmap('/media/poker_raw/flop.npy', mode = 'r', dtype = np.uint8, shape = (self.dat_yyy.shape[0],3))
        z = np.memmap('/media/poker_raw/cartesianFlop.npy', mode = 'r', dtype = np.uint8, shape = (y.shape[0] * xx.shape[0], xx.shape[1] + y.shape[1]), offset = (start) * y.shape[0] * 5)
        xyz = np.memmap('/media/poker_raw/PMAP_Flop.npy', mode = 'r+', dtype = np.uint8, shape = ((self.dat_yyy.shape[0] - self.numbad) * xx.shape[0], self.dat_yyy.shape[1] + xx.shape[1] + 8))
        idx = 0
        j = 0
        for i in z.tolist():
            if(self.contains_duplicates(i)):
                xyz[idx] = i
                idx += 1
                if((j + 1) % y.shape[0] == 0):
                    xyz.flush()
        
        xyz.flush()


    def map_indices(self, arr):
        
        mp = {}    
        c = 0

        if(len(arr[0]) == 5):
            for x in arr.tolist():
                mp[str(x[0]) + str(x[1]) + str(x[2]) + str(x[3]) + str(x[4])] = c
                c += 1
        elif(len(arr[0]) == 4):

            for x in arr.tolist():
                mp[str(x[0]) + str(x[1]) + str(x[2]) + str(x[3])] = c
                c += 1

        else:

            for x in arr.tolist():
                mp[str(x[0]) + str(x[1]) + str(x[2])] = c
                c += 1
        return mp


    def map_indices_by_hand(self, arr):
        mp = {}    
        c = 0
        for x in arr.tolist():
            mp[str(x[0]) + str(x[1])] = c
            c += 1
        return mp



cluster()

