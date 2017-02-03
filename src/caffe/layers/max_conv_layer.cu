#include <vector>

#include "caffe/layers/max_conv_layer.hpp"

namespace caffe {


template <typename Dtype>
void MaxConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n)  {
      this->forward_gpu_max_conv(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_, n);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void MaxConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        const Dtype* weight = this->blobs_[0]->gpu_data();
        Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
        for (int i = 0; i < top.size(); ++i) {
          const Dtype* top_diff = top[i]->gpu_diff();
          // DEBUG:
          max_idx_.to_cpu();
          int *max_mask = max_idx_.mutable_cpu_data(); // 39*176*14*14
          for (int debug_i = 0; debug_i < 14 * 14; debug_i++) {
            LOG(INFO) << max_mask[debug_i] << std::endl;
          }
          // END DEBUG
          // Bias gradient, if necessary.
          if (this->bias_term_ && this->param_propagate_down_[1]) {
            Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
            for (int n = 0; n < this->num_; ++n) {
              this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
            }
          }
          if (this->param_propagate_down_[0] || propagate_down[i]) {
            const Dtype* bottom_data = bottom[i]->gpu_data();
            Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
            for (int n = 0; n < this->num_; ++n) {
              // gradient w.r.t. weight. Note that we will accumulate diffs.
              if (this->param_propagate_down_[0]) {
                this->weight_gpu_max_gemm(bottom_data + n * this->bottom_dim_,
                    top_diff + n * this->top_dim_, weight_diff, n);
              }
              // gradient w.r.t. bottom data, if necessary.
              if (propagate_down[i]) {
                this->backward_gpu_max_gemm(top_diff + n * this->top_dim_, weight,
                    bottom_diff + n * this->bottom_dim_);
              }
            }
          }
        }
      }

INSTANTIATE_LAYER_GPU_FUNCS(MaxConvolutionLayer);

}  // namespace caffe
