// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
//
// build notes:
/*

../build/pytorch_mlir/test.exe -emit=lower-aten -f test/aten_add.mlir > std_add.mlir
mlir-opt --lower-to-llvm std_add.mlir > llvm_add.mlir
../build/pytorch_mlir/test.exe -emit=llvm -f llvm_add.mlir > add.ll
llc add.ll
gcc -c -o add.o add.s
gcc -c -o aten_lib.o src/aten_lib.c
gcc -o file.so -shared aten_lib.o add.o

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct _tensor_t {
    float *d;
    //size_t offset;
    size_t rank[3];
    //size_t stride[3];
} tensor_t;

tensor_t add_forward(tensor_t *a, tensor_t *b, int j)
{
    // ignoring offset and stride here.

    //printf("A: %p, %lu,%lu,%lu\n", a->d, a->rank[0], a->rank[1], a->rank[2]);
    //printf("B: %p, %lu,%lu,%lu\n", b->d, b->rank[0], b->rank[1], b->rank[2]);
    //printf("j: %d\n", j);

    size_t a_size = a->rank[0]*a->rank[1]*a->rank[2];

    tensor_t c;
    c.d = (float*)malloc(a_size*sizeof(float));

    for (int i=0; i<a_size; i++)
        c.d[i] = a->d[i] + b->d[i];

    return c;
}

extern tensor_t graph(tensor_t *, tensor_t *);

int hello(int a)
{
    printf("hello, world %d\n", a);
    return a+42;
}

float* graph_stub(float *a, float *b, int s0, int s1, int s2, int s3)
{
    tensor_t x;
    tensor_t y;
    //printf("%p %p %d %d %d %d\n", a, b, s0, s1, s2, s3);

    x.rank[0] = s0;
    x.rank[1] = s1;
    x.rank[2] = s2;
    size_t x_size = x.rank[0]*x.rank[1]*x.rank[2];
    x.d = a;

    y.rank[0] = s0;
    y.rank[1] = s1;
    y.rank[2] = s2;
    size_t y_size = y.rank[0]*y.rank[1]*y.rank[2];
    y.d = b;

    tensor_t z = graph(&x, &y);

    return z.d;
}
