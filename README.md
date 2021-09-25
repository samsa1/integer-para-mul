
## How to use this parallel interger multiplication code.

Assume DIR is this directory.

1) Download gmp-6.2.1
2) extract gmp-6.2.1.tar.gz and go to gmp-6.2.1
3) patch -i $DIR/configure.patch
4) patch -i $DIR/gmp-h.patch
4) patch -i $DIR/gmp-impl.patch
5) cp $DIR/mul_fft_para.c mpn/generic/mul_fft.c
6) autoreconf -i
7) ./configure --disable-shared --enable-pthread
8) make -j8


## How to use GMP's benchmark

1) compile GMP as presented above
2) cd tune
3) patch -i $DIR/speed.patch
3) patch -i $DIR/speed_pthread.patch
4) make speed
5) ./speed -s 10000000 -n 32 mpn_mul_n

## How to use our benchmark

1) compile GMP as presented above
2) gcc -I. $DIR/test-gmp-para.c .libs/libgmp.a -O3 -lpthread -lhwloc
3) ./a.out -t=32 -mul=2 10000000
    to have the timers with 1, 2, 4, 8, 16 and 32 threads.