#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/dice_loss_new_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
  void DiceLossNewLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Compute the loss (negative log likelihood)
    const int count = bottom[0]->count();
    const int num = bottom[0]->shape(0);
    const int channel = bottom[0]->shape(1);
    const int ndata = bottom[0]->count(2);
  // Stable version of loss computation from input data
    const Dtype* input_data = bottom[0]->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype temp_loss = (Dtype)0;
    int denominator;
    if (behavior&2) {
      Dtype PP=0, GG=0, PG=0;
      caffe_gpu_dot(count, input_data, input_data, &PP);
      caffe_gpu_dot(count, target, target, &GG);
      caffe_gpu_dot(count, input_data, target, &PG);
      temp_loss += (Dtype)(2.0) * PG/(PP + GG);
      denominator = 1;
    } else {
      Dtype PP=0, GG=0, PG=0;
      caffe_gpu_dot(count, input_data, input_data, &PP);
      caffe_gpu_dot(count, target, target, &GG);
      caffe_gpu_dot(count, input_data, target, &PG);
      temp_loss += (Dtype)(2.0) * PG/(PP + GG);
      denominator = 1;
    }
    if (behavior&1) {
      top[0]->mutable_cpu_data()[0]=temp_loss/((Dtype)denominator);
    }
    else{
      top[0]->mutable_cpu_data()[0]=temp_loss;
    }
  }

template <typename Dtype>
__global__ void DiceLossNewBackward(const int n, const Dtype* input, const Dtype* target, Dtype PP, Dtype GG, Dtype PG, const int size, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    bottom_diff[index] =-(Dtype)(2.0) * (target[index] * (PP + GG) - ((Dtype)(2.0) * input[index] * PG))/((Dtype)(size) * (PP + GG) * (PP + GG));
  }
}

template <typename Dtype>
  void DiceLossNewLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
      LOG(FATAL) << this->type()
      << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
    // First, compute the diff
      const int count = bottom[0]->count();
      const int num = bottom[0]->shape(0);
      const int channel = bottom[0]->shape(1);
      const int ndata = bottom[0]->count(2);
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const Dtype* input_data = bottom[0]->gpu_data();
      const Dtype* target = bottom[1]->gpu_data();
      if (behavior&2) {
        Dtype PP = 0, GG = 0, PG = 0;
        caffe_gpu_dot(count, input_data, input_data, &PP);
        caffe_gpu_dot(count, target, target, &GG);
        caffe_gpu_dot(count, input_data, target, &PG);
	DiceLossNewBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, input_data, target, PP, GG, PG, 1, bottom_diff);
      } else {
        Dtype PP = 0, GG = 0, PG = 0;
        caffe_gpu_dot(count, input_data, input_data, &PP);
        caffe_gpu_dot(count, target, target, &GG);
        caffe_gpu_dot(count, input_data, target, &PG);
	DiceLossNewBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, input_data, target, PP, GG, PG, 1, bottom_diff);
      }
    }
  }

INSTANTIATE_LAYER_GPU_FUNCS(DiceLossNewLayer);

}  // namespace caffe
