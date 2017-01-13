#include <algorithm>
#include <vector>
#include <limits>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
#ifndef CPU_ONLY

template <typename Dtype>
__global__ void MaxConvPool(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels, // num = 1, channels = 1 in our case (to save GPU memory)
    const int height, const int width, const int pooled_height, // height = 176*15*15, width = 14 * 14 pooled_height = 176 in our case
    const int pooled_width, const int kernel_h, const int kernel_w, // pooled_width = 14 * 14, kernel_h = 15 * 15, kernel_w = 1 in our case
    const int stride_h, const int stride_w, const int pad_h, const int pad_w, // stride_h = 15 * 15, stride_w = 1, pad_h = pad_w = 0 in our case
    Dtype* const top_data, int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    mask[index] = maxidx;
    top_mask[index] = maxidx;
  }
}

template <typename Dtype>
__global__ void AfterPoolSum(const int nthreads, 
    const Dtype* const bottom_data, const int height,
    const int width, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    top_data[index] = 0;
    for (int i = 0; i < height; ++i) {
        top_data[index] += bottom_data[index + i * width];
    }
  }
}

template <typename Dtype> // TODO (Zhishuai): only valid when group_ == 1
void BaseConvolutionLayer<Dtype>::forward_gpu_max_conv(const Dtype* input,
    const Dtype* weights, Dtype* output, int num_idx, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, this->col_buffer_.mutable_gpu_data());
    }
    col_buff = this->col_buffer_.gpu_data(); // The size will be (176*15*15)*(14)*(14) in our case
  }
//  Dtype *multiply_res = new Dtype[this->blobs_[0]->count(1) * conv_out_spatial_dim_]; // (176*15*15)*(14*14) in our case
  Dtype *dev_multiply_res; 
  Dtype *dev_max_pooled;
  Dtype *dev_summed;
  cudaMalloc((void **) &dev_multiply_res, (this->blobs_[0]->count(1) * conv_out_spatial_dim_) * sizeof(Dtype)); // (176*15*15)*(14*14) in our case
  cudaMalloc((void **) &dev_max_pooled, (this->blobs_[0]->shape(0) * this->blobs_[0]->shape(1) * conv_out_spatial_dim_) * sizeof(Dtype)); // 39*176*(14*14) in our case
  cudaMalloc((void **) &dev_summed, (this->blobs_[0]->shape(0) * conv_out_spatial_dim_) * sizeof(Dtype)); // 39*14*14 in our case
  for (int g = 0; g < this->group_; ++g) {
      for (int im_ = 0; im_ < this->conv_out_channels_; ++im_) { // 39 in our case
          caffe_gpu_dgmm<Dtype>(CblasRight, conv_out_spatial_dim_, this->blobs_[0]->count(1),
            col_buff + col_offset_ * g, weights + weight_offset_ * g + im_ * this->blobs_[0]->count(1),
            1, dev_multiply_res); // get element-wise multiplication, saved in multiply_res
//          cudaMemcpy(multiply_res, dev_multiply_res, (this->blobs_[0]->count(1) * conv_out_spatial_dim_)*sizeof(Dtype),
//            cudaMemcpyDeviceToHost);
          MaxConvPool<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
            dev_multiply_res, 1, 1, // num = 1, channels = 1 in our case (to save GPU memory); channels may be set to be 39 if the GPU memory is sufficient
            this->blobs_[0]->count(1), conv_out_spatial_dim_, this->blobs_[0]->shape(1), // height = 176*15*15, width = 14 * 14 pooled_height = 176 in our case
            conv_out_spatial_dim_, this->blobs_[0]->count(2), 1, // pooled_width = 14 * 14, kernel_h = 15 * 15, kernel_w = 1 in our case
            this->blobs_[0]->count(2), 1, 0, 0, // stride_h = 15 * 15, stride_w = 1, pad_h = pad_w = 0 in our case
            dev_max_pooled + im_ * this->blobs_[0]->shape(1) * conv_out_spatial_dim_, mask + (im_ + num_idx * conv_out_channels_) * this->blobs_[0]->shape(1) * conv_out_spatial_dim_);
          AfterPoolSum<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
            dev_max_pooled + im_ * this->blobs_[0]->shape(1) * conv_out_spatial_dim_, this->blobs_[0]->shape(1), conv_out_spatial_dim_, dev_summed + im_ * conv_out_spatial_dim_);
      }
  }
}

#endif  // !CPU_ONLY

}  // namespace caffe
