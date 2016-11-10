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

    bool write_to(
      const google::protobuf::MessageLite& message,
      google::protobuf::io::ZeroCopyOutputStream* rawOutput
    ) {
      // We create a new coded stream for each message.  Don't worry, this is fast.
      google::protobuf::io::CodedOutputStream output(rawOutput);

      // Write the size.
      const int size = message.ByteSize();

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

      // Tell the stream not to read beyond that size.
      google::protobuf::io::CodedInputStream::Limit limit = input.PushLimit(size);

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
        if (remove(file_name_) != 0) {
          LOG(ERROR) << "Error when deleting file [" << file_name_ << "]";
        }

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

      opened_source_queue_lock_.lock();
      opened_source_queue_.push(std::string(file_name_));
      opened_source_queue_lock_.unlock();
    }

    void PipeCursor::Next() {
      clear_current();

      int error_no = context_->readDelimitedFrom(&current_);

      if (error_no == 1) {
        throw std::runtime_error("Unhandled streaming read");
      } else if (error_no < 0) {
        LOG(ERROR) << "Invalid read " << "@" << fake_key_;
      }

      ++fake_key_;
    }

    void PipeTransaction::Commit() {
      int commit_count = 0;
      boost::uuids::basic_random_generator<boost::mt19937> gen;
      boost::uuids::uuid u = gen();
      std::string uuid_str = boost::uuids::to_string(u);
      stringstream ss;

      ss << MRFeatureExtraction::get_share_memory_fs_path()
         << '/'
         << MRFeatureExtraction::get_from_nn_batch_file_name_prefix()
         << uuid_str;
      std::string file_name = ss.str();
      std::ofstream output_stream(file_name.c_str(), std::ios::binary);
      google::protobuf::io::ZeroCopyOutputStream* raw_output_stream =
        new google::protobuf::io::OstreamOutputStream(&output_stream);

      while(!batch_.empty()) {
        write_delimited_to(batch_.front(), raw_output_stream);

        batch_.pop();
        ++commit_count;
      }

      delete raw_output_stream;
      output_stream.close();

      stringstream buf_ss;
      buf_ss << file_name << "\n";
      std::string buf = buf_ss.str();

      write(out_fd_, buf.c_str(), buf.length()); // Do not write the tailing '\0' to pipe.

      if (commit_count > MRFeatureExtraction::get_batch_size()) {
        throw std::runtime_error("Batch size larger than batch size.");
      }

      opened_source_queue_lock_.lock();
      if (opened_source_queue_.empty()) {
        opened_source_queue_lock_.unlock();

        throw std::runtime_error("Input batch count and output batch count do not match.");
      }

      std::remove(opened_source_queue_.front().c_str());
      opened_source_queue_.pop();
      opened_source_queue_lock_.unlock();
    }

    std::mutex Pipe::opened_source_queue_lock_;
    std::queue<std::string> Pipe::opened_source_queue_;
  }
}

