//
// Created by zen on 26/08/16.
//

#ifndef CAFFE_IMAGE_FOLDER_DATA_LAYER_HPP_
#define CAFFE_IMAGE_FOLDER_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/image_data_layer.hpp"

namespace caffe {

  template<typename Dtype>
  class ImageFolderDataLayer : public ImageDataLayer<Dtype> {
  public:
    explicit ImageFolderDataLayer(const LayerParameter &param)
        : ImageDataLayer<Dtype>(param) {
      current_label_ = 0;
    }

    virtual ~ImageFolderDataLayer();

    virtual inline const char *type() const { return "ImageFolderData"; }

    virtual inline int ExactNumBottomBlobs() const { return 0; }

    virtual inline int ExactNumTopBlobs() const { return 2; }

  protected:
    virtual void load_images();
    virtual void transform_image(
      Batch<Dtype>* batch,
      Dtype* prefetch_data,
      int item_id,
      const cv::Mat& cv_img,
      Blob<Dtype>* transformed_blob
    );
    virtual int get_batch_size();

  private:
    void get_image_files(const std::string &path, std::map<string, int> image_label_map);
    void load_known_labels(const std::string &path);
    void save_debug_known_labels(const std::string &path);
    void load_image_labels(const std::string &path, std::map<std::string, int>& m);
    void save_debug_image_labels(const std::string &path);

    inline bool is_label(const std::string& str) {
      if (str.empty() || str[0] != 'n') {
        return false;
      }

      int i = 1;
      for (; i < str.length(); ++i) {
        if (str[i] < '0' || str[i] > '9') {
          return false;
        }
      }

      return i == 9;
    }

  private:
    int current_label_;
    std::map<std::string, int> image_folder_label_map_;
  };


}  // namespace caffe

#endif //CAFFE_IMAGE_FOLDER_DATA_LAYER_HPP_
