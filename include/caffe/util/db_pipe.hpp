//
// Created by Zengpan Fan on 8/2/16.
//

#ifndef CAFFE_UTIL_DB_PIPE_HPP
#define CAFFE_UTIL_DB_PIPE_HPP

#include <fstream>
#include <queue>
#include <stdexcept>
#include <string>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "leveldb/db.h"
#include "leveldb/write_batch.h"

#include "caffe/util/db.hpp"
#include "caffe/proto/caffe.pb.h"

#include "mr_feature_extraction.hpp"

namespace db=caffe::db;

namespace caffe {
  namespace db {
    class PipeCursor : public Cursor {
    public:
      explicit PipeCursor(std::string& source): current_to_nn_batch_index_(-1),
                                                current_to_nn_batch_fd_(-1),
                                                current_to_nn_batch_file_stream_(NULL) {
        input_stream_.open(source.c_str(), std::ios_base::in | std::ios_base::out);

        Next();
      }
      ~PipeCursor() {
        input_stream_.close();
        delete current_to_nn_batch_file_stream_;
        close(current_to_nn_batch_fd_);
      }
      virtual void SeekToFirst() { } // TODO(zen): use ZeroCopyInputStream::BackUp
      virtual void Next();
      virtual std::string key() {
        return "";
      }
      virtual std::string value() { return current_.SerializeAsString(); }
      virtual bool valid() { return valid_; }

    private:
      bool valid_;
      std::ifstream input_stream_;
      caffe::Datum current_;

      int current_to_nn_batch_index_;
      int current_to_nn_batch_fd_;
      google::protobuf::io::ZeroCopyInputStream *current_to_nn_batch_file_stream_;
    };

    class PipeTransaction : public Transaction {
    public:
      explicit PipeTransaction(std::string& source): current_from_nn_batch_id_(0),
                                                     current_from_nn_batch_fd_(-1) {
        out_stream_.open(source.c_str(), std::ios_base::in | std::ios_base::out);
      }
      virtual void Put(const std::string& key, const std::string& value) {
        batch_.push(value);
      }
      virtual void Commit();
      ~PipeTransaction() {
        out_stream_.close();
      }

    private:
      std::fstream out_stream_;
      google::protobuf::io::ZeroCopyOutputStream* output_;
      std::queue<std::string> batch_;
      caffe::Datum msg_;

      int current_from_nn_batch_id_;
      int current_from_nn_batch_fd_;

    DISABLE_COPY_AND_ASSIGN(PipeTransaction);
    };

    class Pipe : public DB {
    public:
      Pipe() { }
      virtual ~Pipe() { Close(); }
      virtual void Open(const std::string& source, db::Mode mode) {
        source_ = source;
        mode_ = mode;
      }
      virtual void Close() { }
      virtual PipeCursor* NewCursor() {
        if (mode_ != db::READ) {
          std::ostringstream str_stream;
          str_stream << "Can only create cursor on read only pipe";
          throw std::runtime_error(str_stream.str());
        }

        return new PipeCursor(source_);
      }
      virtual PipeTransaction* NewTransaction() {
        if (mode_ != db::WRITE && mode_ != db::NEW) {
          std::ostringstream str_stream;
          str_stream << "Can only create transaction on write only pipe";
          throw std::runtime_error(str_stream.str());
        }

        return new PipeTransaction(source_);
      }

    private:
      std::string source_;
      db::Mode mode_;
    };
  }  // namespace db
}  // namespace caffe


#endif //CAFFE_UTIL_DB_PIPE_HPP
