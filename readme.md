Fun code is in the source folder.

If you modify ccluster.pyx, be sure to rebuild using:
```
python ccluster-helper.py build_ext --inplace
```

To see what uses python, execute the below command and open ccluster.html in the source directory. 
```
cython ccluster.pyx -a
```
Python dense areas will he highlighted yellow.


Currently will generate a (semi-sparce) 50+ gb numpy matrix.

Elements can be accessed using two mappings. One maps your hand to a particular chunk. The other maps the board to an index within the chunk. This scales and will be more intuitive when you see it. Point being that element access is not quite O(1)... but most nearly.



Looking for a kmeans++ clustering algorithm written in c/cython to cluster river, turn, and flop datasets. 
