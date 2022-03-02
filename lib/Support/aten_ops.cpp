//===- aten_ops.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <assert.h>
#include <iostream>
#include <vector>

#ifdef ATEN_OPS_ENABLE_TORCH
#include "torch/csrc/api/include/torch/torch.h"

#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#endif

namespace {

static const bool verbose = true;

template<typename T, int N>
struct tensor_t {
  T *d;
  T *aligned;
  size_t offset;
  size_t shape[N];
  size_t stride[N];

  size_t index(size_t n, size_t channel, size_t row, size_t col) const {
    size_t channels = shape[1];
    size_t height = shape[2];
    size_t width = shape[3];
    return n * height * width * channels + channel * height * width + row * width + col;
  }

  tensor_t() {
    d = aligned = nullptr;
    offset = 0;
    for (int i=0; i<N; i++)
      shape[i] = stride[i] = 0;
  }
};

template<typename T, int N>
std::vector<int64_t> translate_shape(tensor_t<T,N> *t)
{
  std::vector<int64_t> shape;
  for (int i=0; i<N; i++) {
    shape.push_back(t->shape[i]);
    //std::cout << i << " shape " << t->shape[i] << std::endl;
  }
  return shape;
}

template<typename T, int N>
std::vector<int64_t> translate_stride(tensor_t<T,N> *t)
{
  std::vector<int64_t> stride;
  for (int i=0; i<N; i++) {
    stride.push_back(t->stride[i]);
    //std::cout << i << " stride " << t->stride[i] << std::endl;
  }
  return stride;
}

#ifdef ATEN_OPS_ENABLE_TORCH
template<typename T, int N>
at::Tensor to_torch(tensor_t<T,N> *t,
                    const at::TensorOptions &options = at::TensorOptions())
{
  //if (verbose) std::cout << "to_torch\n";
  return torch::from_blob((void*)t->d,
                           translate_shape(t),
                           translate_stride(t),
                           options);
}
#endif

template<typename T> void mm_out(tensor_t<T,2> *a, tensor_t<T,2> *b, tensor_t<T,2> *r);

template<typename T, int N>
void add_out(tensor_t<T,N> *a, tensor_t<T,N> *b, T alpha, tensor_t<T,N> *result)
{
  size_t numel = 1;
  for (size_t d=0; d<N; d++)
    numel *= a->shape[d];

  for (size_t n=0; n<numel; n++)
    result->d[n] = a->d[n] + b->d[n];
}

template<typename T, int N>
void add_out(tensor_t<T,N> *a, T b, T alpha, tensor_t<T,N> *result)
{
  size_t numel = 1;
  for (size_t d=0; d<N; d++)
    numel *= a->shape[d];

  for (size_t n=0; n<numel; n++)
    result->d[n] = a->d[n] + b;
}

template<typename T>
void addmm_out(tensor_t<T,1> *a, tensor_t<T,2> *b, tensor_t<T,2> *c,
               int32_t alpha, int32_t beta, tensor_t<T,2> *r)
{
  mm_out<T>(b,c,r);
  size_t numel = r->shape[0] * r->shape[1];

  for (size_t n=0; n<numel; n++)
    r->d[n] += a->d[n%a->shape[0]];
}

template<typename T, int N, int M>
void as_strided_out(tensor_t<float,M> *a,
           /*size*/ int32_t sz0, int32_t sz1, int32_t sz2, int32_t sz3,
         /*stride*/ int32_t sd0, int32_t sd1, int32_t sd2, int32_t sd3,
                    int32_t offset, tensor_t<T,N> *r)
{
#ifdef ATEN_OPS_ENABLE_TORCH
  at::Tensor input = to_torch(a);

  std::vector<int64_t> size;
  std::vector<int64_t> stride;
  c10::optional<int64_t> storage_offset;

  if (offset != 0) storage_offset = offset;
  if (N > 0) {
    size.push_back(sz0);
    stride.push_back(sd0);
  }
  if (N > 1) {
    size.push_back(sz1);
    stride.push_back(sd1);
  }
  if (N > 2) {
    size.push_back(sz2);
    stride.push_back(sd2);
  }
  if (N > 3) {
    size.push_back(sz3);
    stride.push_back(sd3);
  }

  std::vector<int64_t> sizeRef{size};
  std::vector<int64_t> strideRef{stride};

  //for (int i = 0; i<N; i++)
  //  std::cout << "STRIDE " << i << " " << stride[i] << std::endl;
  at::Tensor result = at::native::as_strided_tensorimpl(input, size, stride, storage_offset).clone();

  memcpy(r->d, result.data_ptr(), result.numel()*sizeof(T));
#else
  std::cout << "aten_ops " << __func__ << "is not enabled (requires PyTorch)\n";
#endif

}

template<typename T>
void acap_conv2d_hw_kernel(tensor_t<T,4> *t, tensor_t<T,4> *weight, tensor_t<T,1> *bias,
                           int32_t stride, int32_t pad, int32_t dilation, tensor_t<T,4> *r,
                           size_t batch_sw, size_t batch_hw, size_t batch_start,
                           size_t ofm_channels_sw, size_t ofm_height_sw, size_t ofm_width_sw, size_t ofm_channels_hw, size_t ofm_channel_start,
                           size_t ifm_channels_sw, size_t ifm_height_sw, size_t ifm_width_sw, size_t ifm_channels_hw, size_t ifm_channel_start,
                           size_t kernel_height, size_t kernel_width,
                           size_t h_start, size_t h_size, size_t w_start, size_t w_size,
                           size_t herd_row, size_t herd_col, size_t virtual_herd_row, size_t virtual_herd_col)
{

  if (1) {
    std::cout << "herd (" << herd_row << ", " << herd_col << ") ";
    //std::cout << "vherd (" << virtual_herd_row << ", " << virtual_herd_col << ") ";
    std::cout << batch_hw << " " << batch_start << " " << batch_sw << ", ";
    std::cout << ifm_channels_hw << " " << ifm_channel_start << " " << ifm_channels_sw << ", ";
    std::cout << ofm_channels_hw << " " << ofm_channel_start << " " << ofm_channels_sw << ", ";
    std::cout << h_size << " " << h_start << " " << ofm_height_sw << ", ";
    std::cout << w_size << " " << w_start << " " << ofm_width_sw << std::endl;
  }

  for (size_t i=0; i<batch_hw; i++) {
    auto n = batch_start + i;
    if (n >= batch_sw) continue;

  for (size_t j=0; j<ofm_channels_hw; j++) {
    auto ofm_channel = ofm_channel_start + j;
    if (ofm_channel >= ofm_channels_sw) continue;

  for (size_t k=0; k<ifm_channels_hw; k++) {
    auto ifm_channel = ifm_channel_start + k;
    if (ifm_channel >= ifm_channels_sw) continue;

    auto w_offset_base = 0;
    if (weight->shape[0] != ofm_channels_sw)
      w_offset_base += (j/*ofm_channel*/ * weight->shape[1] * kernel_height * kernel_width);
    else
      w_offset_base += (ofm_channel * weight->shape[1] * kernel_height * kernel_width);

    if (weight->shape[1] != ifm_channels_sw)
      w_offset_base += (k/*ifm_channel*/ * kernel_height * kernel_width);
    else
      w_offset_base += (ifm_channel * kernel_height * kernel_width);

    for (size_t _ofm_row = 0; _ofm_row < h_size; ++_ofm_row) {
      auto ofm_row = _ofm_row + h_start;
      if (ofm_row >= ofm_height_sw) continue;
      for (size_t _ofm_col = 0; _ofm_col < w_size; ++_ofm_col) {
        auto ofm_col = _ofm_col + w_start;
        if (ofm_col >= ofm_width_sw) continue;

        T accum = 0;
        //if (ifm_channel == 0)
        //  accum = bias->d[ofm_channel];

        for (size_t ky = 0; ky < kernel_height; ++ky) {
          for (size_t kx = 0; kx < kernel_width; ++kx) {
            auto ifm_row = (t->shape[2] != ifm_height_sw) ?
                            (_ofm_row * stride + (ky/* - pad*/)) :
                            (ofm_row * stride + (ky/* - pad*/));
            auto ifm_col = (t->shape[3] != ifm_width_sw) ?
                            (_ofm_col * stride + (kx/* - pad*/)) :
                            (ofm_col * stride + (kx/* - pad*/));
            auto batch = (t->shape[0] != batch_sw) ? i : n;
            auto chan = (t->shape[1] != ifm_channels_sw) ? k : ifm_channel;
            auto x_offset = t->index(batch, chan, ifm_row, ifm_col);
            auto w_offset = w_offset_base + (ky * kernel_width) + kx;
            if (ifm_row >= 0 && ifm_row < ifm_height_sw && ifm_col >= 0 && ifm_col < ifm_width_sw) {
              accum += t->d[x_offset] * weight->d[w_offset];
            }
          }
        }
        auto batch = (r->shape[0] != batch_sw) ? i : n;
        auto chan = (r->shape[1] != ofm_channels_sw) ? j : ofm_channel;
        auto row = (r->shape[2] != ofm_height_sw) ? _ofm_row : ofm_row;
        auto col = (r->shape[3] != ofm_width_sw) ? _ofm_col : ofm_col;
        r->d[r->index(batch, chan, row, col)] += accum;
      }
    }
  }
}
}
}

template<typename T>
void conv2d_out(tensor_t<T,4> *t, tensor_t<T,4> *weight, tensor_t<T,1> *bias,
                int32_t stride, int32_t pad, int32_t dilation, tensor_t<T,4> *r)
{
  // tensor_t<T,4> result;
  // result.shape[0] = t->shape[0];
  // result.shape[1] = weight->shape[0];
  // result.shape[2] = 1 + ((t->shape[2] - weight->shape[2] + 2*pad/*[0]*/) / stride/*[0]*/);
  // result.shape[3] = 1 + ((t->shape[3] - weight->shape[3] + 2*pad/*[1]*/) / stride/*[1]*/);
  // size_t numel = r->shape[0] * r->shape[1] * r->shape[2] * r->shape[3];
  // r->d = r->aligned = (T*)malloc(numel*sizeof(T));

  // from hydratype
  size_t num_tensors = t->shape[0];
  size_t ifm_channels = t->shape[1];
  size_t ifm_height = t->shape[2];
  size_t ifm_width = t->shape[3];
  size_t ofm_channels = r->shape[1];
  size_t ofm_height = r->shape[2];
  size_t ofm_width = r->shape[3];
  size_t kernel = weight->shape[2];

  for (size_t n = 0; n < num_tensors; ++n) {
    for (size_t ofm_channel = 0; ofm_channel < ofm_channels; ++ofm_channel) {
      for (size_t ofm_row = 0; ofm_row < ofm_height; ++ofm_row) {
        for (size_t ofm_col = 0; ofm_col < ofm_width; ++ofm_col) {
          T acc = 0;
          if (verbose)
            std::cout << "batch " << n << " output " << n << "," << ofm_channel << "," << ofm_row << "," << ofm_col  << std::endl;
          for (size_t ifm_channel = 0; ifm_channel < ifm_channels; ++ifm_channel) {
            int w_offset_base = (ofm_channel * ifm_channels * kernel * kernel) + (ifm_channel * kernel * kernel);
            for (size_t ky = 0; ky < kernel; ++ky) {
              for (size_t kx = 0; kx < kernel; ++kx) {
                int ifm_row = (int)ofm_row * stride + (ky - pad);
                int ifm_col = (int)ofm_col * stride + (kx - pad);
                if (ifm_row >= 0 && ifm_row < (int)ifm_height && ifm_col >= 0 && ifm_col < (int)ifm_width) {
                  int x_offset = t->index(n, ifm_channel, ifm_row, ifm_col);
                  int w_offset = w_offset_base + (ky * kernel) + kx;

                  acc += t->d[x_offset] * weight->d[w_offset];
                  if (verbose) {
                    std::cout << "  in " << x_offset << " " << t->d[x_offset] << " "
                              << " weight " << weight->d[w_offset] << " "
                              << " acc " << acc << " " << acc << std::endl;
                  }
                }
                else {
                }
              }
            }
          }
          if (true/*bias*/) {
            acc += bias->d[ofm_channel];
          }
          int y_offset = r->index(n, ofm_channel, ofm_row, ofm_col);
          r->d[y_offset] = acc;
        }
      }
    }
  }
}

template<typename T>
void conv2d_backward_out(tensor_t<T,4> *grad_output, tensor_t<T,4> *input,
                         tensor_t<T,4> *weight, int32_t stride, int32_t pad, int32_t dilation,
                         tensor_t<T,4> *r0, tensor_t<T,4> *r1, tensor_t<T,1> *r2)
{
#ifdef ATEN_OPS_ENABLE_TORCH
  const at::Tensor &arg_grad = to_torch(grad_output);
  const at::Tensor &arg_input = to_torch(input);
  const at::Tensor &arg_weight = to_torch(weight);

  std::vector<int64_t> p{pad,pad};
  std::vector<int64_t> s{stride, stride};
  std::vector<int64_t> d{dilation, dilation};

  std::array<bool, 3> output_mask{true, true, true};

  std::tuple<at::Tensor,at::Tensor,at::Tensor> grads
    = at::native::mkldnn_convolution_backward(arg_input, arg_grad, arg_weight,
                                              p, s, d, 1, output_mask);

  auto result0 = std::get<0>(grads);
  auto result1 = std::get<1>(grads);
  auto result2 = std::get<2>(grads);

  memcpy(r0->d, result0.data_ptr(), result0.numel()*sizeof(T));
  memcpy(r1->d, result1.data_ptr(), result1.numel()*sizeof(T));
  memcpy(r2->d, result2.data_ptr(), result2.numel()*sizeof(T));
#else
  std::cout << "aten_ops " << __func__ << "is not enabled (requires PyTorch)\n";
#endif
}

template<typename T, int N>
void log_softmax_out(tensor_t<T,N> *t, int32_t dim, bool half_to_float, tensor_t<T,N> *r)
{
#ifdef ATEN_OPS_ENABLE_TORCH
  at::Tensor input = to_torch(t);
  at::Tensor result = at::native::log_softmax_cpu(input, dim, half_to_float);
  memcpy(r->d, result.data_ptr(), result.numel()*sizeof(T));
#else
  std::cout << "aten_ops " << __func__ << "is not enabled (requires PyTorch)\n";
#endif

}

template<typename T, int N>
void log_softmax_backward_data_out(tensor_t<T,N> *a, tensor_t<T,N> *b,
                                   int32_t c, tensor_t<T,N> *d, tensor_t<T,N> *r)
{
#ifdef ATEN_OPS_ENABLE_TORCH
  at::Tensor inputA = to_torch(a);
  at::Tensor inputB = to_torch(b);
  at::Tensor inputD = to_torch(d);

  at::Tensor result = at::native::log_softmax_backward_cpu(inputA, inputB, c, inputD);
  memcpy(r->d, result.data_ptr(), result.numel()*sizeof(T));
#else
  std::cout << "aten_ops " << __func__ << "is not enabled (requires PyTorch)\n";
#endif

}

template<typename T>
void max_pool2d_with_indices_out(tensor_t<T,4> *t, int32_t c, int32_t d,
                                 int32_t e, int32_t f, bool ceil_mode,
                                 tensor_t<T,4> *r0, tensor_t<int64_t,4> *r1)
{
#ifdef ATEN_OPS_ENABLE_TORCH
  at::Tensor input = to_torch(t);

  std::vector<int64_t> kernel{c,c};
  std::vector<int64_t> stride{d,d};
  std::vector<int64_t> padding{e,e};
  std::vector<int64_t> dilation{f,f};

  auto result = at::native::max_pool2d_with_indices_cpu(input, kernel, stride,
                                                        padding, dilation, ceil_mode);
  at::Tensor outTensor = std::get<0>(result);
  at::Tensor idxTensor = std::get<1>(result);
  memcpy(r0->d, outTensor.data_ptr(), outTensor.numel()*sizeof(T));
  memcpy(r1->d, idxTensor.data_ptr(), idxTensor.numel()*sizeof(T));
#else
  std::cout << "aten_ops " << __func__ << "is not enabled (requires PyTorch)\n";
#endif

}

template<typename T>
void max_pool2d_with_indices_backward_out(tensor_t<T,4> *a, tensor_t<T,4> *b,
                                          int32_t c, int32_t d, int32_t e, int32_t f,
                                          bool g, tensor_t<int64_t,4> *h, tensor_t<T,4> *r)
{
#ifdef ATEN_OPS_ENABLE_TORCH
  const at::Tensor &inputA = to_torch(a);
  const at::Tensor &inputB = to_torch(b);
  at::TensorOptions options(at::ScalarType::Long);
  const at::Tensor &inputH = to_torch(h, options);

  std::vector<int64_t> kernel{c,c};
  std::vector<int64_t> stride{d,d};
  std::vector<int64_t> padding{e,e};
  std::vector<int64_t> dilation{f,f};

  at::Tensor result = at::native::max_pool2d_with_indices_backward_cpu(inputA, inputB,
                                                                       kernel,
                                                                       stride,
                                                                       padding,
                                                                       dilation,
                                                                       g, inputH);
  memcpy(r->d, result.data_ptr(), result.numel()*sizeof(T));
#else
  std::cout << "aten_ops " << __func__ << "is not enabled (requires PyTorch)\n";
#endif

}

template<typename T>
void mm_out(tensor_t<T,2> *a, tensor_t<T,2> *b, tensor_t<T,2> *r)
{
  size_t a_h = a->shape[0];
  size_t a_w = a->shape[1];
  size_t b_h = b->shape[0];
  size_t b_w = b->shape[1];
  assert(a_w == b_h);

  for (size_t i=0; i<a_h; i++) {
    for (size_t j=0; j<b_w; j++) {
      size_t idx = i*b_w + j;
      r->d[idx] = (T)(0);
      for (size_t k=0; k<a_w; k++) {
        T _a = a->d[i*a_w + k];
        T _b = b->d[k*b_w + j];
        r->d[idx] += _a * _b;
      }
    }
  }
}

template<typename T, int N>
void mul_out(tensor_t<T,N> *a, tensor_t<T,N> *b, tensor_t<T,N> *r)
{
  size_t numel = 1;
  for (size_t d=0; d<N; d++)
    numel *= a->shape[d];

  for (size_t n=0; n<numel; n++)
    r->d[n] = a->d[n] * b->d[n];
}

template<typename T, int N>
void relu_out(tensor_t<T,N> *a, tensor_t<T,N> *r)
{
  size_t numel = 1;
  for (size_t d=0; d<N; d++)
    numel *= a->shape[d];

  for (size_t n=0; n<numel; n++)
    r->d[n] = a->d[n] > 0.0f ? a->d[n] : 0.0f;
}

template<typename T>
void t_out(tensor_t<T,2> *a, tensor_t<T,2> *r)
{
  size_t h = a->shape[0];
  size_t w = a->shape[1];

  for (size_t i=0; i<h; i++)
    for (size_t j=0; j<w; j++)
      r->d[j*h+i] = a->d[i*w+j];
}

template<typename T, int N>
void threshold_backward_out(tensor_t<T,N> *a, tensor_t<T,N> *b, int32_t c, tensor_t<T,N> *r)
{
#ifdef ATEN_OPS_ENABLE_TORCH
  at::Tensor inputA = to_torch(a);
  at::Tensor inputB = to_torch(b);

  at::Tensor result = at::native::threshold_backward(inputA, inputB, c);
  memcpy(r->d, result.data_ptr(), result.numel()*sizeof(T));
#else
  std::cout << "aten_ops " << __func__ << "is not enabled (requires PyTorch)\n";
#endif

}

template<typename T, int N, int M>
void view_out(tensor_t<T,M> *a, int32_t b, int32_t c, int32_t d, int32_t e, tensor_t<T,N> *r)
{
  tensor_t<T,N> result;
  size_t numel = 1;
  for (size_t d=0; d<M; d++)
    numel *= a->shape[d];

  if (N == 1)
    c = d = e = 1;
  if (N == 2)
    d = e = 1;
  if (N == 3)
    e = 1;

  int inferred = 0;
  if (b == -1) inferred++;
  if (c == -1) inferred++;
  if (d == -1) inferred++;
  if (e == -1) inferred++;
  assert(inferred <= 1 && "aten.view Error: only one dimension can be inferred");

  if (b == -1)
    b = numel / (c * d * e);
  if (c == -1)
    c = numel / (b * d * e);
  if (d == -1)
    d = numel / (b * c * e);
  if (e == -1)
    e = numel / (b * c * d);

  if (N > 0)
    r->shape[0] = b;
  if (N > 1)
    r->shape[1] = c;
  if (N > 2)
    r->shape[2] = d;
  if (N > 3)
    r->shape[3] = e;

  memcpy(r->d, a->d, numel*sizeof(T));
}


} // namespace

extern "C" {

// add_out

void add_1F32_1F32_1F32_out(tensor_t<float,1> *a, tensor_t<float,1> *b, int32_t i, tensor_t<float,1> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  add_out<float,1>(a,b,i,r);
}

void add_2F32_2F32_2F32_out(tensor_t<float,2> *a, tensor_t<float,2> *b, int32_t i, tensor_t<float,2> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  add_out<float,2>(a,b,i,r);
}

void add_3F32_3F32_3F32_out(tensor_t<float,3> *a, tensor_t<float,3> *b, int32_t i, tensor_t<float,3> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  add_out<float,3>(a,b,i,r);
}

void add_4F32_4F32_4F32_out(tensor_t<float,4> *a, tensor_t<float,4> *b, int32_t i, tensor_t<float,4> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  add_out<float,4>(a,b,i,r);
}

void add_4I32_4I32_4I32_out(tensor_t<int32_t,4> *a, tensor_t<int32_t,4> *b, int32_t i, tensor_t<int32_t,4> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  add_out<int32_t,4>(a,b,i,r);
}

void add_2F32_2F32_out(tensor_t<float,2> *a, float f, int32_t i, tensor_t<float,2> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  add_out<float,2>(a,f,i,r);
}

// addmm_out

void addmm_2F32_1F32_2F32_2F32_out(tensor_t<float,1> *a, tensor_t<float,2> *b,
                                   tensor_t<float,2> *c, int32_t alpha, int32_t beta,
                                   tensor_t<float,2> *r) {
if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  addmm_out<float>(a,b,c,alpha,beta,r);
}

// as_strided_out

void as_strided_1F32_1F32_out(tensor_t<float,1> *a,
                    /*size*/  int32_t sz0, int32_t sz1, int32_t sz2, int32_t sz3,
                  /*stride*/  int32_t sd0, int32_t sd1, int32_t sd2, int32_t sd3,
                              int32_t offset, tensor_t<float,1> *r)
{
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  as_strided_out<float,1,1>(a, sz0, sz1, sz2, sz3, sd0, sd1, sd2, sd3, offset, r);
}

void as_strided_4F32_2F32_out(tensor_t<float,2> *a,
                    /*size*/  int32_t sz0, int32_t sz1, int32_t sz2, int32_t sz3,
                  /*stride*/  int32_t sd0, int32_t sd1, int32_t sd2, int32_t sd3,
                              int32_t offset, tensor_t<float,4> *r)
{
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  as_strided_out<float,4,2>(a, sz0, sz1, sz2, sz3, sd0, sd1, sd2, sd3, offset, r);
}

// conv2d_out

void conv2d_4F32_4F32_4F32_1F32_out(tensor_t<float,4> *t, tensor_t<float,4> *weight,
                                    tensor_t<float,1> *bias, int32_t stride,
                                    int32_t padding, int32_t dilation, tensor_t<float,4> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  conv2d_out<float>(t, weight, bias, stride, padding, dilation, r);
}

void conv2d_relu_4F32_4F32_4F32_1F32_out(tensor_t<float,4> *t, tensor_t<float,4> *weight,
                                         tensor_t<float,1> *bias, int32_t stride,
                                         int32_t padding, int32_t dilation, tensor_t<float,4> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  conv2d_out<float>(t, weight, bias, stride, padding, dilation, r);
  relu_out<float,4>(r, r);
}

// acap_conv2d_hw_kernel

void acap_conv2d_hw_kernel_4F32_4F32_1F32_4F32_t(tensor_t<float,4> *t, tensor_t<float,4> *weight, tensor_t<float,1> *bias,
                                               int32_t stride, int32_t pad, int32_t dilation, tensor_t<float,4> *r,
                                               size_t batch_sw, size_t batch_hw, size_t batch_start,
                                               size_t ofm_channels_sw, size_t ofm_height_sw, size_t ofm_width_sw, size_t ofm_channels_hw, size_t ofm_channel_start,
                                               size_t ifm_channels_sw, size_t ifm_height_sw, size_t ifm_width_sw, size_t ifm_channels_hw, size_t ifm_channel_start,
                                               size_t kernel_height, size_t kernel_width,
                                               size_t x_offset, size_t x_end, size_t y_offset, size_t y_end,
                                               size_t herd_row, size_t herd_col, size_t virtual_herd_row, size_t virtual_herd_col)
{
    acap_conv2d_hw_kernel<float>(t, weight, bias, stride, pad, dilation, r,
                                 batch_sw, batch_hw, batch_start,
                                 ofm_channels_sw, ofm_height_sw, ofm_width_sw, ofm_channels_hw, ofm_channel_start,
                                 ifm_channels_sw, ifm_height_sw, ifm_width_sw, ifm_channels_hw, ifm_channel_start,
                                 kernel_height, kernel_width,
                                 x_offset, x_end, y_offset, y_end,
                                 herd_row, herd_col, virtual_herd_row, virtual_herd_col);
}

// conv2d_backward_out

void
conv2d_backward_4F32_4F32_1F32_4F32_4F32_4F32_out(tensor_t<float,4> *grad_output, tensor_t<float,4> *t,
                                                  tensor_t<float,4> *weight, int32_t stride,
                                                  int32_t padding, int32_t dilation,
                                                  tensor_t<float,4> *r0, tensor_t<float,4> *r1, tensor_t<float,1> *r2)
{
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  conv2d_backward_out<float>(grad_output, t, weight, stride, padding, dilation, r0, r1, r2);
}

// div
float *div_0F32_0F32_0F32(float *a, float *b)
{
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  float *ret = (float*)malloc(sizeof(float));
  *ret = *a / *b;
  return ret;
}

// log_softmax_out

void log_softmax_1F32_1F32_out(tensor_t<float,1> *t, int32_t dim, bool half_to_float, tensor_t<float,1> *r)
{
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  log_softmax_out<float,1>(t, dim, half_to_float, r);
}
void log_softmax_2F32_2F32_out(tensor_t<float,2> *t, int32_t dim, bool half_to_float, tensor_t<float,2> *r)
{
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  log_softmax_out<float,2>(t, dim, half_to_float, r);
}
void log_softmax_3F32_3F32_out(tensor_t<float,3> *t, int32_t dim, bool half_to_float, tensor_t<float,3> *r)
{
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  log_softmax_out<float,3>(t, dim, half_to_float, r);
}
void log_softmax_4F32_4F32_out(tensor_t<float,4> *t, int32_t dim, bool half_to_float, tensor_t<float,4> *r)
{
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  log_softmax_out<float,4>(t, dim, half_to_float, r);
}

// log_softmax_backward_data_out

void log_softmax_backward_data_2F32_2F32_2F32_2F32_out(tensor_t<float,2> *a, tensor_t<float,2> *b,
                                                   int32_t c, tensor_t<float,2> *d, tensor_t<float,2> *r)
{
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  log_softmax_backward_data_out<float,2>(a, b, c, d, r);
}

void log_softmax_backward_data_4F32_4F32_4F32_4F32_out(tensor_t<float,4> *a, tensor_t<float,4> *b,
                                                   int32_t c, tensor_t<float,4> *d, tensor_t<float,4> *r)
{
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  log_softmax_backward_data_out<float,4>(a, b, c, d, r);
}

// max_pool2d_out

void max_pool2d_with_indices_4F32_4I64_4F32_out(tensor_t<float,4> *t, int32_t kernel, int32_t pad,
                                                int32_t stride, int32_t dilation, bool ceil_mode,
                                                tensor_t<float,4> *r0, tensor_t<int64_t,4> *r1)
{
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  max_pool2d_with_indices_out<float>(t, kernel, pad, stride, dilation, ceil_mode, r0, r1);
}

// max_pool2d backward_out

void max_pool2d_with_indices_backward_4F32_4F32_4F32_4I64_out(tensor_t<float,4> *a, tensor_t<float,4> *b,
                                                              int32_t c, int32_t d, int32_t e, int32_t f,
                                                              bool g, tensor_t<int64_t,4> *h, tensor_t<float,4> *r)
{
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  max_pool2d_with_indices_backward_out<float>(a, b, c, d, e, f, g, h, r);
}

// mm_out

void mm_2F32_2F32_2F32_out(tensor_t<float,2> *a, tensor_t<float,2> *b, tensor_t<float,2> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  mm_out<float>(a,b,r);
}

// tensor_t<float,2> mm_2F32_2F32_2F32(tensor_t<float,2> *a, tensor_t<float,2> *b) {
//   if (verbose) std::cout << "aten_ops " << __func__ << "\n";
//   tensor_t<float,2> r;
//   r.shape[0] = a->shape[0];
//   r.shape[1] = b->shape[1];
//   r.d = r.aligned = (float*)malloc(sizeof(float)*r.shape[0]*r.shape[1]);
//   mm_out<float>(a,b,&r);
//   return r;
// }

// mul_out

void mul_1I1_1I1_1I1_out(tensor_t<bool,1> *a, tensor_t<bool,1> *b, tensor_t<bool,1> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  mul_out<bool,1>(a,b,r);
}

void mul_1F32_1F32_1F32_out(tensor_t<float,1> *a, tensor_t<float,1> *b, tensor_t<float,1> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  mul_out<float,1>(a,b,r);
}

void mul_2F32_2F32_2F32_out(tensor_t<float,2> *a, tensor_t<float,2> *b, tensor_t<float,2> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  mul_out<float,2>(a,b,r);
}

void mul_3F32_3F32_3F32_out(tensor_t<float,3> *a, tensor_t<float,3> *b, tensor_t<float,3> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  mul_out<float,3>(a,b,r);
}

void mul_4F32_4F32_4F32_out(tensor_t<float,4> *a, tensor_t<float,4> *b, tensor_t<float,4> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  mul_out<float,4>(a,b,r);
}

// nll_loss2d_forward_out

void
nll_loss2d_forward_1F32_1F32_4F32_3I64_1F32_out(tensor_t<float,4> *a, tensor_t<uint64_t,3> *b,
                                                tensor_t<float,1> *c, int64_t d, int64_t e,
                                                tensor_t<float,1> *r0, tensor_t<float,1> *r1)
{
#ifdef ATEN_OPS_ENABLE_TORCH
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  using T = float;
  at::Tensor inputA = to_torch(a); 
  at::TensorOptions options(at::ScalarType::Long);
  at::Tensor inputB = to_torch(b, options);
  at::Tensor inputC = to_torch(c);

  std::tuple<at::Tensor,at::Tensor> result =
    at::nll_loss2d_forward(inputA, inputB, inputC, d, e);

  at::Tensor result0 = std::get<0>(result);
  at::Tensor result1 = std::get<1>(result);
  memcpy(r0->d, result0.data_ptr(), result0.numel()*sizeof(T));
  memcpy(r1->d, result1.data_ptr(), result1.numel()*sizeof(T));
#else
  std::cout << "aten_ops " << __func__ << "is not enabled (requires PyTorch)\n";
#endif
}

// nll_loss2d_backward_out

void
nll_loss2d_backward_4F32_1F32_4F32_3I64_1F32_1F32_out(tensor_t<float,1> *a, tensor_t<float,4> *b,
                                                      tensor_t<uint64_t,3> *c, tensor_t<float,1> *d,
                                                      int32_t e, int32_t f, tensor_t<float,1> *g,
                                                      tensor_t<float,4> *r)
{
#ifdef ATEN_OPS_ENABLE_TORCH
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  using T = float;
  at::Tensor inputA = to_torch(a);
  at::Tensor inputB = to_torch(b);
  at::TensorOptions options(at::ScalarType::Long);
  at::Tensor inputC = to_torch(c, options);
  at::Tensor inputD = to_torch(d);
  at::Tensor inputG = to_torch(g);

  at::Tensor result = at::nll_loss2d_backward(inputA, inputB, inputC,
                                              inputD, e, f, inputG);
  memcpy(r->d, result.data_ptr(), result.numel()*sizeof(T));
#else
  std::cout << "aten_ops " << __func__ << "is not enabled (requires PyTorch)\n";
#endif

}

void
nll_loss_backward_2F32_1F32_2F32_1I64_1F32_1F32_out(tensor_t<float,1> *a, tensor_t<float,2> *b,
                                                    tensor_t<uint64_t,1> *c, tensor_t<float,1> *d,
                                                    int32_t e, int32_t f, tensor_t<float,1> *g,
                                                    tensor_t<float,2> *r)
{
#ifdef ATEN_OPS_ENABLE_TORCH
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  using T = float;
  at::Tensor inputA = to_torch(a);
  at::Tensor inputB = to_torch(b);
  at::TensorOptions options(at::ScalarType::Long);
  at::Tensor inputC = to_torch(c, options);
  at::Tensor inputD = to_torch(d);
  at::Tensor inputG = to_torch(g);

  at::Tensor result= at::nll_loss_backward(inputA, inputB, inputC,
                                                     inputD, e, f, inputG);

  memcpy(r->d, result.data_ptr(), result.numel()*sizeof(T));
#else
  std::cout << "aten_ops " << __func__ << "is not enabled (requires PyTorch)\n";
#endif

}

// nll_loss_forward_out

void
nll_loss_forward_1F32_1F32_2F32_1I64_1F32_out(tensor_t<float,2> *a, tensor_t<uint64_t,1> *b,
                                              tensor_t<float,1> *c, int64_t d, int64_t e,
                                              tensor_t<float,1> *r0, tensor_t<float,1> *r1)
{
#ifdef ATEN_OPS_ENABLE_TORCH
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  using T = float;
  at::Tensor inputA = to_torch(a);
  at::TensorOptions options(at::ScalarType::Long);
  at::Tensor inputB = to_torch(b, options);
  at::Tensor inputC = to_torch(c);

  std::tuple<at::Tensor,at::Tensor> result =
    at::nll_loss_forward(inputA, inputB, inputC, d, e);

  at::Tensor result0 = std::get<0>(result);
  at::Tensor result1 = std::get<1>(result);
  
  memcpy(r0->d, result0.data_ptr(), result0.numel()*sizeof(T));
  memcpy(r1->d, result1.data_ptr(), result1.numel()*sizeof(T));
#else
  std::cout << "aten_ops " << __func__ << "is not enabled (requires PyTorch)\n";
#endif

}

// relu_out

void relu_1F32_1F32_out(tensor_t<float,1> *a, tensor_t<float,1> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  relu_out<float,1>(a,r);
}

void relu_2F32_2F32_out(tensor_t<float,2> *a, tensor_t<float,2> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  relu_out<float,2>(a,r);
}

void relu_3F32_3F32_out(tensor_t<float,3> *a, tensor_t<float,3> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  relu_out<float,3>(a,r);
}

void relu_4F32_4F32_out(tensor_t<float,4> *a, tensor_t<float,4> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  relu_out<float,4>(a,r);
}

// t_out

void t_2F32_2F32_out(tensor_t<float,2> *a, tensor_t<float,2> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  t_out<float>(a,r);
}

// threshold_backward_out

void threshold_backward_1F32_1F32_1F32_out(tensor_t<float,1> *a, tensor_t<float,1> *b, int32_t c, tensor_t <float,1> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  threshold_backward_out<float,1>(a,b,c,r);
}

void threshold_backward_2F32_2F32_2F32_out(tensor_t<float,2> *a, tensor_t<float,2> *b, int32_t c, tensor_t <float,2> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  threshold_backward_out<float,2>(a,b,c,r);
}

void threshold_backward_3F32_3F32_3F32_out(tensor_t<float,3> *a, tensor_t<float,3> *b, int32_t c, tensor_t <float,3> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  threshold_backward_out<float,3>(a,b,c,r);
}

void threshold_backward_4F32_4F32_4F32_out(tensor_t<float,4> *a, tensor_t<float,4> *b, int32_t c, tensor_t <float,4> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  threshold_backward_out<float,4>(a,b,c,r);
}

// view_out

void view_1F32_4F32_out(tensor_t<float,4> *a, int32_t b, int32_t c,int32_t d,int32_t e, tensor_t<float,1> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  view_out<float,1,4>(a,b,c,d,e,r);
}

void view_1F32_3F32_out(tensor_t<float,3> *a, int32_t b, int32_t c,int32_t d,int32_t e, tensor_t<float,1> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  view_out<float,1,3>(a,b,c,d,e,r);
}

void view_1F32_2F32_out(tensor_t<float,2> *a, int32_t b, int32_t c,int32_t d,int32_t e, tensor_t<float,1> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  view_out<float,1,2>(a,b,c,d,e,r);
}

void view_2F32_4F32_out(tensor_t<float,4> *a, int32_t b, int32_t c,int32_t d,int32_t e, tensor_t<float,2> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  view_out<float,2,4>(a,b,c,d,e,r);
}

void view_4F32_1F32_out(tensor_t<float,1> *a, int32_t b, int32_t c,int32_t d,int32_t e, tensor_t<float,4> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  view_out<float,4,1>(a,b,c,d,e,r);
}

void view_4F32_2F32_out(tensor_t<float,2> *a, int32_t b, int32_t c,int32_t d,int32_t e, tensor_t<float,4> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  view_out<float,4,2>(a,b,c,d,e,r);
}

void view_4F32_3F32_out(tensor_t<float,3> *a, int32_t b, int32_t c,int32_t d,int32_t e, tensor_t<float,4> *r) {
  if (verbose) std::cout << "aten_ops " << __func__ << "\n";
  view_out<float,4,3>(a,b,c,d,e,r);
}

}
