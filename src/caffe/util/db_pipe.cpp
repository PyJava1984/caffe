//
// Created by Zengpan Fan on 8/2/16.
//
#include <string>
#include <cstdlib>
#include <fcntl.h>
#include "caffe/util/db_pipe.hpp"

namespace db=caffe::db;

namespace caffe {
  namespace db {
    void PipeCursor::Next() {
      if (current_to_nn_batch_index_ == MRFeatureExtraction::get_batch_size() - 1) {
        std::string file_name;
        std::getline(input_stream_, file_name);

        current_to_nn_batch_index_ = -1;

        if (current_to_nn_batch_file_stream_ != NULL) {
          delete current_to_nn_batch_file_stream_;
        }

        if (current_to_nn_batch_fd_ != -1) {
          close(current_to_nn_batch_fd_);
        }

        current_to_nn_batch_fd_ = open(file_name.c_str(), O_RDONLY);
        current_to_nn_batch_file_stream_ =
            new google::protobuf::io::FileInputStream(current_to_nn_batch_fd_);
      }

      valid_ = current_.ParseFromZeroCopyStream(current_to_nn_batch_file_stream_);
      if (!valid_) {
        current_to_nn_batch_index_ = MRFeatureExtraction::get_batch_size() - 1;
      } else {
        ++current_to_nn_batch_index_;
      }
    }

    void PipeTransaction::Commit() {
      int commit_count = 0;
      stringstream ss;

      ss << MRFeatureExtraction::get_share_memory_fs_path()
         << '/'
         << MRFeatureExtraction::get_from_nn_batch_file_name_prefix()
         << current_from_nn_batch_id_;
      std::string file_name = ss.str();
      int fd = open(file_name.c_str(), O_WRONLY | O_CREAT);
      google::protobuf::io::ZeroCopyOutputStream* output = new google::protobuf::io::FileOutputStream(fd);

      while(!batch_.empty()) {
        string value = batch_.front();
        batch_.pop();

        // TODO(zen): remove overhead
        msg_.ParseFromString(value);
        msg_.SerializeToZeroCopyStream(output);

        ++commit_count;
      }

      delete output;
      close(fd);

      out_stream_ << file_name << "\n";

      if (commit_count > MRFeatureExtraction::get_batch_size()) {
        throw std::runtime_error("Batch size larger than batch size.");
      }

      ++current_from_nn_batch_id_;
    }
  }
}