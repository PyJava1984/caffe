//
// Created by Zengpan Fan on 8/2/16.
//

#ifndef CAFFE_UTIL_DB_PIPE_HPP
#define CAFFE_UTIL_DB_PIPE_HPP

#include <string>
#include <queue>
#include <boost/functional/hash.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "leveldb/db.h"
#include "leveldb/write_batch.h"

#include "caffe/util/db.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/foursquare.pb.h"

namespace db=caffe::db;

namespace caffe {
  namespace db {
    class PipeCursor : public Cursor {
    public:
      explicit PipeCursor(int in_pipe) {
        input_ = new google::protobuf::io::FileInputStream(in_pipe);

        const void* buffer;
        int size;

        valid_ = input_->Next(&buffer, &size);
      }
      ~PipeCursor() {
        delete input_;
      }
      virtual void SeekToFirst() { }
      virtual void Next() {
        const void* buffer;
        int size;

        input_->Next(&buffer, &size);
        current_.ParseFromZeroCopyStream(input_);
      }
      virtual std::string key() {
        std::stringstream str_stream;

        str_stream << key_hash(current_.serializedmessage());

        return str_stream.str();
      }
      virtual std::string value() { return current_.serializedmessage(); }
      virtual bool valid() { return valid_; }

    private:
      bool valid_;
      boost::hash<std::string> key_hash;
      google::protobuf::io::ZeroCopyInputStream* input_;
      caffe::SerializedMessage current_;
    };

    class PipeTransaction : public Transaction {
    public:
      explicit PipeTransaction(int out_pipe) : out_pipe_(out_pipe) { }
      virtual void Put(const std::string& key, const std::string& value) {
        batch_.push(value);
      }
      virtual void Commit() {
        while(!batch_.empty()) {
          string value = batch_.front();
          batch_.pop();

          msg_.clear_serializedmessage();
          msg_.set_serializedmessage(value);
          msg_.SerializeToZeroCopyStream(output_);
        }
      }

    private:
      int out_pipe_;
      google::protobuf::io::ZeroCopyOutputStream* output_;
      std::queue<std::string> batch_;
      caffe::SerializedMessage msg_;

    DISABLE_COPY_AND_ASSIGN(PipeTransaction);
    };

    class Pipe : public DB {
    public:
      Pipe() : pipe_(-1) { }
      virtual ~Pipe() { Close(); }
      virtual void Open(const std::string& source, db::Mode mode);
      virtual void Close() {
        if (pipe_ != -1) {
          close(pipe_);
          pipe_ = -1;
        }

      }
      virtual PipeCursor* NewCursor() {
        if (mode_ != db::READ) {
          std::ostringstream str_stream;
          str_stream << "Can only create cursor on read only pipe";
          throw std::runtime_error(str_stream.str());
        }

        return new PipeCursor(pipe_);
      }
      virtual PipeTransaction* NewTransaction() {
        if (mode_ != db::WRITE && mode_ != db::NEW) {
          std::ostringstream str_stream;
          str_stream << "Can only create transaction on write only pipe";
          throw std::runtime_error(str_stream.str());
        }

        return new PipeTransaction(pipe_);
      }

    private:
      int pipe_;
      db::Mode mode_;
    };
  }  // namespace db
}  // namespace caffe


#endif //CAFFE_UTIL_DB_PIPE_HPP
