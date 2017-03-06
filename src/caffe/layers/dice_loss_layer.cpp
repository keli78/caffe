#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/dice_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void DiceLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  int dim = bottom[0]->count() / bottom[0]->num();
  this->PG = (Dtype *)malloc(bottom[0]->count()*sizeof(Dtype));
  this->PP = (Dtype *)malloc(bottom[0]->count()*sizeof(Dtype));
  this->GG = (Dtype *)malloc(bottom[0]->count()*sizeof(Dtype));
  this->SPG = (Dtype *)malloc(dim*sizeof(Dtype));
  this->SPP = (Dtype *)malloc(dim*sizeof(Dtype));
  this->SGG = (Dtype *)malloc(dim*sizeof(Dtype));
  this->SPP_GG = (Dtype *)malloc(dim*sizeof(Dtype));
  this->D = (Dtype *)malloc(dim*sizeof(Dtype));
}

template <typename Dtype>
void DiceLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "DICE_LOSS layer inputs must have the same count.";
  free(this->PG);
  this->PG = (Dtype *)malloc(bottom[0]->count()*sizeof(Dtype));
  free(this->PP);
  this->PP = (Dtype *)malloc(bottom[0]->count()*sizeof(Dtype));
  free(this->GG);
  this->GG = (Dtype *)malloc(bottom[0]->count()*sizeof(Dtype));
  free(this->SPG);
  this->SPG = (Dtype *)malloc(dim*sizeof(Dtype));
  free(this->SPP);
  this->SPP = (Dtype *)malloc(dim*sizeof(Dtype));
  free(this->SGG);
  this->SGG = (Dtype *)malloc(dim*sizeof(Dtype));
  free(this->SPP_GG);
  this->SPP_GG = (Dtype *)malloc(dim*sizeof(Dtype));
  free(this->D);
  this->D = (Dtype *)malloc(dim*sizeof(Dtype));
}

template <typename Dtype>
void DiceLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Compute the loss (negative log likelihood)
  //const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  caffe_mul(bottom[0]->count(),input_data,target, this->PG);
  caffe_powx(bottom[0]->count(), input_data, (Dtype)2., this->PP);
  caffe_powx(bottom[0]->count(), target, (Dtype)2., this->GG);

  for (int i  = 0; i < dim; ++i) {
    this->SPG[i] = caffe_cpu_stride_asum(num, this->PG+i, dim);
    this->SPP[i] = caffe_cpu_stride_asum(num, this->PP+i, dim);
    this->SGG[i] = caffe_cpu_stride_asum(num, this->GG+i, dim);
    this->SPP_GG[i] = this->SPP[i]+this->SGG[i];
    this->D[i]=2*this->SPG[i]/this->SPP_GG[i];
  }

  top[0]->mutable_cpu_data()[0]=caffe_cpu_asum(dim,this->D);
}

template <typename Dtype>
void DiceLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* input_data = bottom[0]->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    int dim = bottom[0]->count() / bottom[0]->num();
    Dtype* tmp_data1 = (Dtype *)malloc(dim*sizeof(Dtype));
    Dtype* tmp_data2 = (Dtype *)malloc(dim*sizeof(Dtype));
    Dtype* tmp_data3 = (Dtype *)malloc(dim*sizeof(Dtype));
    for (int i = 0; i < num; ++i) {
      caffe_mul(dim, this->SPP_GG, target + i * dim, tmp_data1);
      caffe_mul(dim, this->SPG, input_data + i * dim, tmp_data2);
      caffe_cpu_scale(dim, (Dtype)2., tmp_data2, tmp_data3);
      caffe_sub(dim, tmp_data1, tmp_data3, tmp_data2);
      caffe_powx(dim, this->SPP_GG, (Dtype)2., tmp_data1);
      caffe_div(dim, tmp_data2, tmp_data1, tmp_data3);
      caffe_cpu_scale(dim, (Dtype)2., tmp_data3, bottom_diff+i*dim);
    }
    free(tmp_data1);
    free(tmp_data2);
    free(tmp_data3);
  }
}

#ifdef CPU_ONLY
STUB_GPU(DiceLossLayer);
#endif

INSTANTIATE_CLASS(DiceLossLayer);
REGISTER_LAYER_CLASS(DiceLoss);

}  // namespace caffe
