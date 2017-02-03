#include <algorithm>
#include <vector>
#include <limits>
#include <float.h>

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
  }
}

template <typename Dtype>
__global__ void MaxPoolMaskApply(const int nthreads,
    Dtype* data, const int num, const int channels, // num = 1, channels = 1 in our case (to save GPU memory)
    const int height, const int width, const int pooled_height, // height = 176*15*15, width = 14 * 14 pooled_height = 176 in our case
    const int pooled_width, const int kernel_h, const int kernel_w, // pooled_width = 14 * 14, kernel_h = 15 * 15, kernel_w = 1 in our case
    const int stride_h, const int stride_w, const int pad_h, const int pad_w, // stride_h = 15 * 15, stride_w = 1, pad_h = pad_w = 0 in our case
    const int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (h * width + w != mask[index]) {
          data[h * width + w] = 0;
        }
      }
    }
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
  //Dtype *multiply_res = new Dtype[this->blobs_[0]->count(1) * conv_out_spatial_dim_]; // (176*15*15)*(14*14) in our case
  Dtype *dev_multiply_res;
  Dtype *dev_max_pooled;
  //Dtype *dev_summed;
  int* mask = max_idx_.mutable_gpu_data();
  int count = this->blobs_[0]->shape(1) * conv_out_spatial_dim_;
  CUDA_CHECK(cudaMalloc((void **) &dev_multiply_res, (this->blobs_[0]->count(1) * conv_out_spatial_dim_) * sizeof(Dtype))); // (176*15*15)*(14*14) in our case
  CUDA_CHECK(cudaMalloc((void **) &dev_max_pooled, (this->blobs_[0]->shape(0) * this->blobs_[0]->shape(1) * conv_out_spatial_dim_) * sizeof(Dtype))); // 39*176*(14*14) in our case
  //cudaMalloc((void **) &dev_summed, (this->blobs_[0]->shape(0) * conv_out_spatial_dim_) * sizeof(Dtype)); // 39*14*14 in our case
  for (int g = 0; g < this->group_; ++g) {
      for (int im_ = 0; im_ < this->conv_out_channels_; ++im_) { // 39 in our case
          caffe_gpu_dgmm<Dtype>(CblasRight, conv_out_spatial_dim_, this->blobs_[0]->count(1),
            col_buff + col_offset_ * g, weights + weight_offset_ * g + im_ * this->blobs_[0]->count(1),
            1, dev_multiply_res); // get element-wise multiplication, saved in multiply_res
          MaxConvPool<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
            dev_multiply_res, 1, 1, // num = 1, channels = 1 in our case (to save GPU memory); channels may be set to be 39 if the GPU memory is sufficient
            this->blobs_[0]->count(1), conv_out_spatial_dim_, this->blobs_[0]->shape(1), // height = 176*15*15, width = 14 * 14 pooled_height = 176 in our case
            conv_out_spatial_dim_, this->blobs_[0]->count(2), 1, // pooled_width = 14 * 14, kernel_h = 15 * 15, kernel_w = 1 in our case
            this->blobs_[0]->count(2), 1, 0, 0, // stride_h = 15 * 15, stride_w = 1, pad_h = pad_w = 0 in our case
            dev_max_pooled + im_ * this->blobs_[0]->shape(1) * conv_out_spatial_dim_, mask + (im_ + num_idx * conv_out_channels_) * this->blobs_[0]->shape(1) * conv_out_spatial_dim_);
          AfterPoolSum<<<CAFFE_GET_BLOCKS(conv_out_spatial_dim_), CAFFE_CUDA_NUM_THREADS>>>(conv_out_spatial_dim_,
            dev_max_pooled + im_ * this->blobs_[0]->shape(1) * conv_out_spatial_dim_, this->blobs_[0]->shape(1), conv_out_spatial_dim_, output + im_ * conv_out_spatial_dim_);
      }
  }
  CUDA_CHECK(cudaFree(dev_multiply_res));
  CUDA_CHECK(cudaFree(dev_max_pooled));
  CUDA_POST_KERNEL_CHECK;
  // DEBUG:
  int *max_gpu_mask = max_idx_.mutable_gpu_data(); // 39*176*14*14
  int *max_cpu_mask = malloc(39*176*14*14*sizeof(Dype));
  CUDA_CHECK(cudaMemcpy(max_cpu_mask, max_gpu_mask, 39*176*14*14*sizeof(Dype), cudaMemcpyDeviceToHost));
  for (int debug_i = 0; debug_i < 14 * 14; debug_i++) {
    LOG(INFO) << max_cpu_mask[debug_i] << std::endl;
  }
  // END DEBUG
}


template <typename Dtype> // TODO (Zhishuai): only valid when group_ == 1
void BaseConvolutionLayer<Dtype>::weight_gpu_max_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights, int num_idx) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  Dtype *col_buff_masked;
  int count = this->blobs_[0]->shape(1) * conv_out_spatial_dim_;
  int* mask = max_idx_.mutable_gpu_data();
  CUDA_CHECK(cudaMalloc((void **) &col_buff_masked, col_buffer_.count(0) * sizeof(Dtype)));
  for (int g = 0; g < group_; ++g) {
    for (int im_ = 0; im_ < this->conv_out_channels_; ++im_) { // 39 in our case
      CUDA_CHECK(cudaMemcpy(col_buff_masked, col_buff, col_buffer_.count(0) * sizeof(Dtype), cudaMemcpyDeviceToDevice));
      MaxPoolMaskApply<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
        col_buff_masked, 1, 1, // num = 1, channels = 1 in our case (to save GPU memory); channels may be set to be 39 if the GPU memory is sufficient
        this->blobs_[0]->count(1), conv_out_spatial_dim_, this->blobs_[0]->shape(1), // height = 176*15*15, width = 14 * 14 pooled_height = 176 in our case
        conv_out_spatial_dim_, this->blobs_[0]->count(2), 1, // pooled_width = 14 * 14, kernel_h = 15 * 15, kernel_w = 1 in our case
        this->blobs_[0]->count(2), 1, 0, 0, // stride_h = 15 * 15, stride_w = 1, pad_h = pad_w = 0 in our case
        mask + (im_ + num_idx * conv_out_channels_) * this->blobs_[0]->shape(1) * conv_out_spatial_dim_);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g + im_ * conv_out_spatial_dim_, col_buff_masked + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g + im_ * this->blobs_[0]->count(1));
      }
  }
  CUDA_CHECK(cudaFree(col_buff_masked));
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_max_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  bool reset_ = true;
  Dtype* data_diff_masked;
  CUDA_CHECK(cudaMalloc((void **) &data_diff_masked, col_buffer_.count(0) * sizeof(Dtype)));
  for (int g = 0; g < group_; ++g) {
    for (int im_ = 0; im_ < this->conv_out_channels_; ++im_) { // 39 in our case
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g + im_ * this->blobs_[0]->count(1),
        output + output_offset_ * g + im_ * conv_out_spatial_dim_,
        (Dtype)0. , data_diff_masked);
        if (reset_ == true) {
          CUDA_CHECK(cudaMemcpy(col_buff, data_diff_masked, col_buffer_.count(0) * sizeof(Dtype), cudaMemcpyDeviceToDevice));
        }
        else {
          caffe_gpu_axpy<Dtype>(col_buffer_.count(0), (Dtype)1., data_diff_masked, col_buff);
        }
      reset_ = false;
      }
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
  CUDA_CHECK(cudaFree(data_diff_masked));
}


template
void BaseConvolutionLayer<float>::backward_gpu_max_gemm(const float* output,
    const float* weights, float* input);

template
void BaseConvolutionLayer<double>::backward_gpu_max_gemm(const double* output,
    const double* weights, double* input);


template
void BaseConvolutionLayer<double>::forward_gpu_max_conv(const double* input,
    const double* weights, double* output, int num_idx, bool skip_im2col);

template
void BaseConvolutionLayer<float>::forward_gpu_max_conv(const float* input,
    const float* weights, float* output, int num_idx, bool skip_im2col);

template
void BaseConvolutionLayer<float>::weight_gpu_max_gemm(const float* input,
    const float* output, float* weights, int num_idx);

template
void BaseConvolutionLayer<double>::weight_gpu_max_gemm(const double* input,
    const double* output, double* weights, int num_idx);

#endif  // !CPU_ONLY

}  // namespace caffe
