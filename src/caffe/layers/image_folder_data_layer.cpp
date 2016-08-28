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
  void ImageFolderDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                                   const vector<Blob<Dtype> *> &top) {
    const int new_height = this->layer_param_.image_data_param().new_height();
    const int new_width = this->layer_param_.image_data_param().new_width();
    const bool is_color = this->layer_param_.image_data_param().is_color();
    std::string root_folder = this->layer_param_.image_data_param().root_folder();

    CHECK((new_height == 0 && new_width == 0) ||
          (new_height > 0 && new_width > 0)) << "Current implementation requires "
        "new_height and new_width to be set at the same time.";
    // Read the file with filenames and labels
    const string &source = this->layer_param_.image_data_param().source();
    LOG(INFO) << "Opening folder " << source;

    fs::path absolute_path = fs::canonical(root_folder, source);
    get_image_files(absolute_path.string());

    CHECK(!this->lines_.empty()) << "File is empty";

    if (this->layer_param_.image_data_param().shuffle()) {
      // randomly shuffle data
      LOG(INFO) << "Shuffling data";
      const unsigned int prefetch_rng_seed = caffe_rng_rand();
      this->prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
      this->ShuffleImages();
    }
    LOG(INFO) << "A total of " << this->lines_.size() << " images.";

    this->lines_id_ = 0;
    // Check if we would need to randomly skip a few data points
    if (this->layer_param_.image_data_param().rand_skip()) {
      unsigned int skip = caffe_rng_rand() %
                          this->layer_param_.image_data_param().rand_skip();
      LOG(INFO) << "Skipping first " << skip << " data points.";
      CHECK_GT(this->lines_.size(), skip) << "Not enough points to skip";
      this->lines_id_ = skip;
    }
    // Read an image, and use it to initialize the top blob.
    cv::Mat cv_img = ReadImageToCVMat(root_folder + this->lines_[this->lines_id_].first,
                                      new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << this->lines_[this->lines_id_].first;
    // Use data_transformer to infer the expected blob shape from a cv_image.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
    this->transformed_data_.Reshape(top_shape);
    // Reshape prefetch_data and top[0] according to the batch_size.
    const int batch_size = this->layer_param_.image_data_param().batch_size();
    CHECK_GT(batch_size, 0) << "Positive batch size required";
    top_shape[0] = batch_size;
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(top_shape);
    }
    top[0]->Reshape(top_shape);

    LOG(INFO) << "output data size: " << top[0]->num() << ","
              << top[0]->channels() << "," << top[0]->height() << ","
              << top[0]->width();
    // label
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }

  INSTANTIATE_CLASS(ImageFolderDataLayer);

  REGISTER_LAYER_CLASS(ImageFolderData);

}  // namespace caffe
#endif  // USE_OPENCV
