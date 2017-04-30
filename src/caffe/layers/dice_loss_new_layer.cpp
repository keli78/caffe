#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/dice_loss_new_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void DiceLossNewLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void DiceLossNewLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "DICE_LOSS layer inputs must have the same count.";
}

template <typename Dtype>
void DiceLossNewLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Compute the loss (negative log likelihood)
  //const int count = bottom[0]->count();
  const int num = bottom[0]->shape(0);
  const int channel = bottom[0]->shape(1);
  const int ndata = bottom[0]->count(2);
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype temp_loss = (Dtype)0;
  for (int n=0; n < num; ++n)
    for (int c=0; c< channel;++c) {
      Dtype PP=0;
      Dtype GG=0;
      Dtype PG=0;
      int start_index=n*ndata*channel+c*ndata;
      for (int i=0; i<ndata;++i) {
        PP+=input_data[start_index+i]*input_data[start_index+i];
        GG+=target[start_index+i]*target[start_index+i];
        PG+=input_data[start_index+i]*target[start_index+i];
      }
      temp_loss+=(Dtype)(2.0)*PG/(PP+GG);
    }
  top[0]->mutable_cpu_data()[0]=temp_loss/((Dtype)(num*channel));
}

template <typename Dtype>
void DiceLossNewLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int num = bottom[0]->shape(0);
    const int channel = bottom[0]->shape(1);
    const int ndata = bottom[0]->count(2);
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* input_data = bottom[0]->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    for (int n=0; n<num; ++n)
      for (int c=0; c<channel; ++c) {
        Dtype PP=0;
        Dtype GG=0;
        Dtype PG=0;
        int start_index=n*ndata*channel+c*ndata;
        for (int i=0; i< ndata; ++i) {
          PP+=input_data[start_index+i]*input_data[start_index+i];
          GG+=target[start_index+i]*target[start_index+i];
          PG+=input_data[start_index+i]*target[start_index+i];
        }
        for (int i=0; i<ndata;++i) {
          bottom_diff[start_index+i]=-(Dtype)(2.0)*
            (target[start_index+i]*(PP+QQ)-((Dtype)(2.0))*input_data[start_index+i]*PQ)/
            (((Dtype)(num*channel))*(PP+QQ)*(PP+QQ));
        }
      }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DiceLossNewLayer);
#endif

INSTANTIATE_CLASS(DiceLossNewLayer);
REGISTER_LAYER_CLASS(DiceLoss);

}  // namespace caffe
