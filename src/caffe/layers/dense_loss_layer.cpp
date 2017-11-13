#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layers/dense_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/*template <typename Dtype>
Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2) {
  Dtype left = std::max(x1 - w1 / 2, x2 - w2 / 2);
  Dtype right = std::min(x1 + w1 / 2, x2 + w2 / 2);
  return right - left;
}

template <typename Dtype>
Dtype Calc_iou(const vector<Dtype>& box, const vector<Dtype>& truth) {
  Dtype w = Overlap(box[0], box[2], truth[0], truth[2]);
  Dtype h = Overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;
  Dtype inter_area = w * h;
  Dtype union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area / union_area;
}

template <typename Dtype>
Dtype Calc_rmse(const vector<Dtype>& box, const vector<Dtype>& truth) {
  return sqrt(pow(box[0]-truth[0], 2) +
              pow(box[1]-truth[1], 2) +
              pow(box[2]-truth[2], 2) +
              pow(box[3]-truth[3], 2));
              }*/

template <typename Dtype>
void DenseLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  /*DetectionLossParameter param = this->layer_param_.detection_loss_param();
  side_ = param.side();
  num_class_ = param.num_class();
  num_object_ = param.num_object();
  sqrt_ = param.sqrt();
  constriant_ = param.constriant();
  object_scale_ = param.object_scale();
  noobject_scale_ = param.noobject_scale();
  class_scale_ = param.class_scale();
  coord_scale_ = param.coord_scale();*/
  
  //int input_count = bottom[0]->count(1);
  //int label_count = bottom[1]->count(1);
  // outputs: classes, iou, coordinates
  //int tmp_input_count = side_ * side_ * (num_class_ + (1 + 4) * num_object_);
  // label: isobj, class_label, coordinates
  //int tmp_label_count = side_ * side_ * (1 + 1 + 1 + 4);
  //CHECK_EQ(input_count, tmp_input_count);
  //CHECK_EQ(label_count, tmp_label_count);
}

template <typename Dtype>
void DenseLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  diff_.ReshapeLike(*bottom[0]);
  
  ones_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  for (int i = 0; i < bottom[0]->count(); ++i) {
    ones_.mutable_cpu_data()[i] = Dtype(1);
  }
}

template <typename Dtype>
void DenseLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* diff = diff_.mutable_cpu_data();
  vector<int> b0_shape = bottom[0]->shape();
  vector<int> b1_shape = bottom[1]->shape();
  Dtype loss(0.0);
  Dtype dif(0.0);
  caffe_set(diff_.count(), Dtype(0.), diff);
  int batch_size = bottom[0]->num();
   int dims = bottom[0]->count(1);
   
   for(int k=0;k < bottom[0]->num();k++)
  {
    int index = k * bottom[0]->count(1);

    //int true_index = k * bottom[1]->count(1);
    for(int i=0;i < dims;i ++)
    {
      //dif =  pow(input_data[i] - (label_data[i]/1000),2);
      int pindex = index + i;
      //int ptrue_index= true_index + i;
      float hhh = (float)label_data[pindex]/10;
      
      dif = input_data[pindex] - (label_data[pindex]/100);
      diff[pindex] =dif;
      loss += dif*dif;
    }
    
    }
   //Dtype dot = caffe_cpu_dot(batch_size*dims, diff_.cpu_data(), diff.cpu_data());
  top[0]->mutable_cpu_data()[0] = loss/batch_size/dims;
  // LOG(INFO) << "average objects: " << obj_count;
  LOG(INFO) << "loss: " << top[0]->mutable_cpu_data()[0];
  //LOG(INFO) << "dot: " << dot;
}

template <typename Dtype>
void DenseLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype sign(1.);
    const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();
    caffe_cpu_axpby(
        bottom[0]->count(),
        alpha,
        diff_.cpu_data(),
        Dtype(0),
        bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(DenseLossLayer);
#endif

INSTANTIATE_CLASS(DenseLossLayer);
REGISTER_LAYER_CLASS(DenseLoss);

}  // namespace caffe




