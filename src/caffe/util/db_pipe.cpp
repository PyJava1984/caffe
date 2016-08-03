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
    void Pipe::Open(const std::string& source, db::Mode mode) {
      switch (mode) {
        case db::READ:
          pipe_ = ::open(source.c_str(), O_RDONLY);
          break;
        case db::WRITE:
          pipe_ = ::open(source.c_str(), O_WRONLY);
          break;
        case db::NEW:
          pipe_ = ::open(source.c_str(), O_CREAT | O_WRONLY);
          break;
        default:
          std::ostringstream str_stream;
          str_stream << "Unknown open mode [" << mode << "]";
          throw std::runtime_error(str_stream.str());
      }

      if (pipe_ == -1) {
        std::ostringstream str_stream;
        str_stream << "Failed to open pipe [" << source << "] with error code [" << errno << "]";
        throw std::runtime_error(str_stream.str());
      }

      mode_ = mode;
    }
  }
}