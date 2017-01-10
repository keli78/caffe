#include <vector>

#include "caffe/layers/max_conv_layer.hpp"

namespace caffe {
    
template <typename Dtype>
void MaxConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top) { // TODO(Zhishuai): We assume group = 1 here
      const int first_spatial_axis = channel_axis_ + 1;
      CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
          << "bottom num_axes may not change.";
      num_ = bottom[0]->count(0, channel_axis_);
      CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
          << "Input size incompatible with convolution kernel.";
      // TODO: generalize to handle inputs of different shapes.
      for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
        CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
            << "All inputs must have the same shape.";
      }
      // Shape the tops.
      bottom_shape_ = &bottom[0]->shape();
      compute_output_shape();
      vector<int> top_shape(bottom[0]->shape().begin(),
          bottom[0]->shape().begin() + channel_axis_);
      top_shape.push_back(num_output_);
      for (int i = 0; i < num_spatial_axes_; ++i) {
        top_shape.push_back(output_shape_[i]);
      }
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        top[top_id]->Reshape(top_shape);
      }
      if (reverse_dimensions()) {
        conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
      } else {
        conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
      }
      col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
      output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
      // Setup input dimensions (conv_input_shape_).
      vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
      conv_input_shape_.Reshape(bottom_dim_blob_shape);
      int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
      for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
        if (reverse_dimensions()) {
          conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
        } else {
          conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
        }
      }
      // The im2col result buffer will only hold one image at a time to avoid
      // overly large memory usage. In the special case of 1x1 convolution
      // it goes lazily unused to save memory.
      col_buffer_shape_.clear();
      col_buffer_shape_.push_back(kernel_dim_ * group_);
      for (int i = 0; i < num_spatial_axes_; ++i) {
        if (reverse_dimensions()) {
          col_buffer_shape_.push_back(input_shape(i + 1));
        } else {
          col_buffer_shape_.push_back(output_shape_[i]);
        }
      }
      col_buffer_.Reshape(col_buffer_shape_);
      bottom_dim_ = bottom[0]->count(channel_axis_);
      top_dim_ = top[0]->count(channel_axis_);
      num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
      num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
      // Set up the all ones "bias multiplier" for adding biases by BLAS
      out_spatial_dim_ = top[0]->count(first_spatial_axis);
      if (bias_term_) {
        vector<int> bias_multiplier_shape(1, out_spatial_dim_);
        bias_multiplier_.Reshape(bias_multiplier_shape);
        caffe_set(bias_multiplier_.count(), Dtype(1),
            bias_multiplier_.mutable_cpu_data());
      }
      max_idx_.Reshape(bottom[0]->num(), num_output_, channels_, 
        conv_out_spatial_dim_); // BatchSize*39*176*(14*14) in our case
}

template <typename Dtype> // Copied from conv layer code
void MaxConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void MaxConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void MaxConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      forward_cpu_max_conv(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_, n);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype> // TODO (Zhishuai): only valid when group_ == 1
void MaxConvolutionLayer<Dtype>::forward_cpu_max_conv(const Dtype* input,
    const Dtype* weights, Dtype* output, int num_idx, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data(); // The size will be 1*(176*15*15)*(14)*(14) in our case
  }
  Dtype* transposed_col_buff_ = new Dtype[this->blobs_[0]->count(1)]; // 176*15*15 in our case
  Dtype* dot_proc_ = new Dtype[this->blobs_[0]->count(1)]; // 176*15*15 in our case
  Dtype* max_mask = max_idx_.mutable_cpu_data();
  for (int g = 0; g < group_; ++g) {
      for (int im_ = 0; im_ < conv_out_channels_; ++im_) { // 39 in our case
          for (int in_ = 0; in_ < conv_out_spatial_dim_; ++in_) { // 14^2 in our case
              for (int ic_ = 0; ic_ < this->blobs_[0]->count(1); ++ic_) { // 176*15*15 in our case
                  transposed_col_buff_[ic_] = *(col_buff + col_offset_ * g + in_ + ic_ * conv_out_spatial_dim_); // Transposing col in col_buff into row in transposed_col_buff_
              }
              caffe_mul(this->blobs_[0]->count(1), 
                  weights + weight_offset_ * g + im_ * this->blobs_[0]->count(1), 
                  transposed_col_buff_, 
                  dot_proc_);
              for (int ic_ = 0; ic_ < this->blobs_[0]->shape(1); ++ic_) { // 176 in our case
                  int max_idx = -1;
                  Dtype max_val = -std::numeric_limits<Dtype>::infinity();
                  for (int max_idx_ = ic_ * this->blobs_[0]->count(2); max_idx_ < (ic_ + 1) * this->blobs_[0]->count(2); ++max_idx_) {
                      if (dot_proc_[max_idx_] > max_val) {
                          max_val = dot_proc_[max_idx_];
                          max_idx = max_idx_ - ic_ * this->blobs_[0]->count(2);
                      }
                  }
                  output[in_ + conv_out_spatial_dim_ * im_] = max_val;
                  max_mask[in_ + conv_out_spatial_dim_ * (ic_ + this->blobs_[0]->shape(1) * (im_ + conv_out_channels_ * num_idx))] = max_idx;
              }
          }
      }
  }
  delete[] dot_proc_;
  delete[] transposed_col_buff_;
}

#ifdef CPU_ONLY
STUB_GPU(MaxConvolutionLayer);
#endif

INSTANTIATE_CLASS(MaxConvolutionLayer);

}  // namespace caffe
