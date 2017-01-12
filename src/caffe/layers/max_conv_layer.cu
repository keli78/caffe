#include <vector>

#include "caffe/layers/max_conv_layer.hpp"

namespace caffe {
    

template <typename Dtype>
void MaxConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_max_conv(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_, n);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MaxConvolutionLayer);

}  // namespace caffe
