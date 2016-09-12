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

  template <typename Dtype>
  int ImageFolderDataLayer<Dtype>::get_batch_size() {
    // Reshape prefetch_data and top[0] according to the batch_size.
    const int batch_size = this->layer_param_.image_data_param().batch_size();
    CHECK_GT(batch_size, 0) << "Positive batch size required";
    return batch_size * this->data_transformer_->GetTotalNumber();
  }

  template<typename Dtype>
  void ImageFolderDataLayer<Dtype>::get_image_files(const std::string &path, std::map<string, int> image_label_map) {
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
            if (image_label_map.find(last_folder_name) == image_label_map.end()) {
              LOG(ERROR) << "Add missing label [" << last_folder_name << ']';

              image_folder_label_map_.insert(
                std::make_pair(last_folder_name, current_label_++)
              );
            }
          }
        } else {
          fs::path parent_folder = cp.parent_path();
          fs::path file_name = cp.filename();
          std::string last_folder_name = parent_folder.string().substr(
            parent_folder.string().find_last_of("/\\") + 1,
            parent_folder.string().length() - (parent_folder.string().find_last_of("/\\") + 1)
          );

          if (is_label(last_folder_name)) {
            if (image_folder_label_map_.find(last_folder_name) != image_folder_label_map_.end()) {
              int label = image_folder_label_map_[last_folder_name];

              this->lines_.push_back(std::make_pair(cp.string(), label));
            } else {
              LOG(FATAL) << "Unknown label " << last_folder_name << " for file " << cp;
            }
          }
        }
      }
    }
  }

  template<typename Dtype>
  void ImageFolderDataLayer<Dtype>::load_known_labels(const std::string &path) {
    std::ifstream label_stream(path);

    for(std::string line; std::getline(label_stream, line, '\n'); ) {
      std::istringstream line_stream(line);
      std::string key;
      std::string value;

      std::getline(line_stream, key, '\t');
      std::getline(line_stream, value, '\t');

      int label = std::atoi(value.c_str());

      image_folder_label_map_.insert(
        std::make_pair(key, label)
      );

      LOG(ERROR) << "Add known label name [" << key << "] as label [" << label << "]";

      current_label_ = std::max(label, current_label_);
    }

    current_label_++;
  }

  template<typename Dtype>
  void ImageFolderDataLayer<Dtype>::save_debug_known_labels(const std::string &path) {
    std::ofstream os(path);

    for (std::map<std::string, int>::iterator iter = image_folder_label_map_.begin();
         iter != image_folder_label_map_.end(); ++iter) {
      os << iter->first << '\t' << iter->second << std::endl;
    }

    os.close();
  }

  template<typename Dtype>
  void ImageFolderDataLayer<Dtype>::load_image_labels(const std::string &path, std::map<std::string, int>& m) {
    std::ifstream label_stream(path);

    for(std::string line; std::getline(label_stream, line, '\n'); ) {
      std::istringstream line_stream(line);
      std::string key;
      std::string value;

      std::getline(line_stream, key, ' ');
      std::getline(line_stream, value, ' ');

      int label = std::atoi(value.c_str());

      LOG(ERROR) << "Add known label name [" << key << "] as label [" << label << "]";
      m.insert(std::make_pair(key, label));
    }
  }

  template<typename Dtype>
  void ImageFolderDataLayer<Dtype>::save_debug_image_labels(const std::string &path) {
    std::ofstream os(path);

    for (std::vector<std::pair<std::string, int>>::iterator iter = this->lines_.begin();
         iter != this->lines_.end(); ++iter) {
      os << iter->first << '\t' << iter->second << std::endl;
    }

    os.close();
  }

  template<typename Dtype>
  void ImageFolderDataLayer<Dtype>::load_images() {
    const string &known_label_path = this->layer_param_.image_data_param().known_label_path();
    if (!known_label_path.empty()) {
      fs::path absolute_known_label_path = fs::canonical(known_label_path, this->root_folder_);

      LOG(INFO) << "Load known labels from [" << absolute_known_label_path << ']';
      load_known_labels(absolute_known_label_path.string());
    }

    const string &debug_known_label_path = this->layer_param_.image_data_param().debug_known_label_path();
    if (!debug_known_label_path.empty()) {
      fs::path absolute_debug_known_label_path = fs::complete(debug_known_label_path, this->root_folder_);

      LOG(INFO) << "Save debug known labels to [" << absolute_debug_known_label_path << ']';
      save_debug_known_labels(absolute_debug_known_label_path.string());
    }

    const string &image_label_path = this->layer_param_.image_data_param().image_label_path();
    std::map<string, int> image_label_map;

    if (!image_label_path.empty()) {
      fs::path absolute_image_label_path = fs::canonical(image_label_path, this->root_folder_);

      LOG(INFO) << "Load image labels from [" << absolute_image_label_path << ']';
      load_image_labels(absolute_image_label_path.string(), image_label_map);
    }

    const string &source = this->layer_param_.image_data_param().source();
    LOG(INFO) << "Opening folder " << source;

    fs::path absolute_path = fs::canonical(source, this->root_folder_);
    get_image_files(absolute_path.string(), image_label_map);

    CHECK(!this->lines_.empty()) << "File is empty";

    const string &debug_image_label_path = this->layer_param_.image_data_param().debug_image_label_path();

    if (!debug_image_label_path.empty()) {
      fs::path absolute_debug_image_label_path = fs::complete(debug_image_label_path, this->root_folder_);

      LOG(INFO) << "Save debug image labels to [" << absolute_debug_image_label_path << ']';
      save_debug_image_labels(absolute_debug_image_label_path.string());
    }
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
