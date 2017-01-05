#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/normalization_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NormalizationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//  const NormalizationParameter& normalization_param = this->layer_param_.normalization_param();
}

template <typename Dtype>
void NormalizationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  squared_.Reshape(bottom[0]->num(), bottom[0]->channels(), 
    bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void NormalizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* squared_data = squared_.mutable_cpu_data();
  int n = bottom[0]->shape(0);
  int d = bottom[0]->shape(1); // do the normalization across all channels, so within this dim
  int h = bottom[0]->shape(2);
  int w = bottom[0]->shape(3);
  caffe_sqr<Dtype>(bottom[0]->count(), bottom_data, squared_data);
  for (int in = 0; in < n; ++in) {
	  for (int ih = 0; ih < h; ++ih) {
		  for (int iw = 0; iw < w; ++iw) {
			  Dtype normsqr = caffe_cpu_stride_asum<Dtype>(d, squared_data+in*(d*h*w)+ih*w+iw,h*w);
			  caffe_cpu_stride_scale<Dtype>(d, pow(normsqr, -0.5), bottom_data+in*(d*h*w)+ih*w+iw,
					  top_data+in*(d*h*w)+ih*w+iw,h*w);
		  }
	  }
  }
}

template <typename Dtype>
void NormalizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    return; // Actually not implemented yet
}

INSTANTIATE_CLASS(NormalizationLayer);
REGISTER_LAYER_CLASS(Normalization);


}  // namespace caffe

