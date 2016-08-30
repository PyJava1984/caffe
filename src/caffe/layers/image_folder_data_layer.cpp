#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)

#include <boost/regex.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_folder_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
  namespace fs = boost::filesystem;

  template<typename Dtype>
  ImageFolderDataLayer<Dtype>::~ImageFolderDataLayer<Dtype>() {
    this->StopInternalThread();
  }

  template<typename Dtype>
  void ImageFolderDataLayer<Dtype>::get_image_files(const std::string &path) {
    current_label_ = 0;
    image_folder_label_map_.clear();

    if (!path.empty()) {
      fs::path apk_path(path);
      fs::recursive_directory_iterator end;

      for (fs::recursive_directory_iterator i(apk_path); i != end; ++i) {
        const fs::path cp = (*i);

        if (fs::is_directory(i->status())) {
          std::string last_folder_name = cp.string().substr(
              cp.string().find_last_of("/\\") + 1,
              cp.string().length() - (cp.string().find_last_of("/\\") + 1)
          );

          if (is_label(last_folder_name)) {
            if (image_folder_label_map_.find(last_folder_name) !=
                image_folder_label_map_.end()) {
              throw std::runtime_error("Duplicated label name " + last_folder_name);
            }

            image_folder_label_map_.insert(
                std::make_pair(last_folder_name, current_label_++)
            );
          }
        } else {
          fs::path parent_folder = cp.parent_path();
          std::string last_folder_name = parent_folder.string().substr(
              parent_folder.string().find_last_of("/\\") + 1,
              parent_folder.string().length() - (parent_folder.string().find_last_of("/\\") + 1)
          );

          if (is_label(last_folder_name)) {
            int label = image_folder_label_map_[last_folder_name];
            this->lines_.push_back(std::make_pair(cp.string(), label));
          }
        }
      }
    }
  }

  template<typename Dtype>
  void ImageFolderDataLayer<Dtype>::load_images() {
    const string &source = this->layer_param_.image_data_param().source();
    LOG(INFO) << "Opening folder " << source;

    fs::path absolute_path = fs::canonical(this->root_folder_, source);
    get_image_files(absolute_path.string());

    CHECK(!this->lines_.empty()) << "File is empty";
  }

  template <typename Dtype>
  void ImageFolderDataLayer<Dtype>::transform_image(
      Batch<Dtype>* batch,
      Dtype* prefetch_data,
      int item_id,
      const cv::Mat& cv_img,
      Blob<Dtype>* transformed_blob
  ) {
    // Apply transformations (mirror, crop...) to the image
    int total_transformed_data_number = this->data_transformer_->GetTotalNumber();
    int offset = batch->data_.offset(item_id * total_transformed_data_number);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->MultiTransforms(cv_img, &(this->transformed_data_));
  }

  INSTANTIATE_CLASS(ImageFolderDataLayer);

  REGISTER_LAYER_CLASS(ImageFolderData);

}  // namespace caffe
#endif  // USE_OPENCV
