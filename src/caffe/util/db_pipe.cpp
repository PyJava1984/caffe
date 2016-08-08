//
// Created by Zengpan Fan on 8/2/16.
//
#include <fstream>
#include <fcntl.h>
#include "caffe/util/db_pipe.hpp"

namespace db=caffe::db;

namespace caffe {
  namespace db {
    bool write_delimited_to(
      const google::protobuf::MessageLite& message,
      google::protobuf::io::ZeroCopyOutputStream* rawOutput
    ) {
      // We create a new coded stream for each message.  Don't worry, this is fast.
      google::protobuf::io::CodedOutputStream output(rawOutput);

      // Write the size.
      const int size = message.ByteSize();
      output.WriteVarint32(size);

      uint8_t* buffer = output.GetDirectBufferForNBytesAndAdvance(size);
      if (buffer != NULL) {
        // Optimization:  The message fits in one buffer, so use the faster
        // direct-to-array serialization path.
        message.SerializeWithCachedSizesToArray(buffer);
      } else {
        // Slightly-slower path when the message is multiple buffers.
        message.SerializeWithCachedSizes(&output);
        if (output.HadError()) return false;
      }

      return true;
    }

    int read_delimited_from(
      google::protobuf::io::ZeroCopyInputStream* rawInput,
      google::protobuf::MessageLite* message
    ) {
      // We create a new coded stream for each message.  Don't worry, this is fast,
      // and it makes sure the 64MB total size limit is imposed per-message rather
      // than on the whole stream.  (See the CodedInputStream interface for more
      // info on this limit.)
      google::protobuf::io::CodedInputStream input(rawInput);

      // Read the size.
      uint32_t size;
      if (!input.ReadVarint32(&size)) return 1; // eof

      LOG(ERROR) << "msg size " << size;

      // Tell the stream not to read beyond that size.
      google::protobuf::io::CodedInputStream::Limit limit =
          input.PushLimit(size);

      // Parse the message.
      if (!message->MergeFromCodedStream(&input)) return -1;
      if (!input.ConsumedEntireMessage()) return -1;

      // Release the limit.
      input.PopLimit(limit);

      return 0;
    }

    std::atomic<long> PipeCursor::fake_key_(0l);

    // Will delete the previous file
    void PipeReadContext::open_to_nn_batch_stream() {
      current_to_nn_batch_stream_lock_.lock();

      if (error_no_ == 0) {
        return;
      }

      if (current_to_nn_batch_stream_ != NULL) {
        delete current_to_nn_batch_stream_;
      }

      if (file_name_ != NULL) {
        std::remove(file_name_);
        free(file_name_);
      }

      size_t nbytes = 255;
      file_name_ = (char *)malloc(nbytes + 1);

      LOG(ERROR) << "Trying to get file name";
      ::getline(&file_name_, &nbytes, input_file_);

      if (current_to_nn_batch_fd_ != -1) {
        close(current_to_nn_batch_fd_);
      }

      file_name_[strlen(file_name_) - 1] = '\0';

      LOG(ERROR) << "Opening to nn batch file [" << file_name_ << "]";

      current_to_nn_batch_fd_ = open(file_name_, O_RDONLY);

      current_to_nn_batch_stream_ =
          new google::protobuf::io::FileInputStream(current_to_nn_batch_fd_);

      error_no_ = 0;

      current_to_nn_batch_stream_lock_.unlock();
    }

    void PipeCursor::Next() {
      int error_no = context_->readDelimitedFrom(&current_);

      if (error_no == 0) {
        LOG(ERROR) << "Channles " << current_.channels();
      } else if (error_no == 1) {
        throw std::runtime_error("Should never reach here");
      } else {
        LOG(ERROR) << "Invalid read " << "@" << fake_key_;
      }
    }

    std::atomic<long> PipeTransaction::current_from_nn_batch_id_(0l);

    void PipeTransaction::Commit() {
      int commit_count = 0;
      stringstream ss;

      ss << MRFeatureExtraction::get_share_memory_fs_path()
         << '/'
         << MRFeatureExtraction::get_from_nn_batch_file_name_prefix()
         << current_from_nn_batch_id_;
      std::string file_name = ss.str();
      int output_fd = open(file_name.c_str(), O_RDWR | O_CREAT);
      google::protobuf::io::ZeroCopyOutputStream* output_stream =
        new google::protobuf::io::FileOutputStream(output_fd);

      while(!batch_.empty()) {
        const caffe::Datum& msg = batch_.front();

        write_delimited_to(msg, output_stream);

        batch_.pop();
        ++commit_count;
      }

      close(output_fd);
      delete output_stream;

      stringstream buf_ss;
      buf_ss << file_name << "\n";
      std::string buf = buf_ss.str();

      write(out_fd_, buf.c_str(), buf.length() + 1);

      if (commit_count > MRFeatureExtraction::get_batch_size()) {
        throw std::runtime_error("Batch size larger than batch size.");
      }

      ++current_from_nn_batch_id_;
    }
  }
}
