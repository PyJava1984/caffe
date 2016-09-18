#ifndef CAFFE_UTIL_DB_HPP
#define CAFFE_UTIL_DB_HPP

#include <string>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe { namespace db {

bool write_delimited_to(
  const google::protobuf::MessageLite& message,
  google::protobuf::io::ZeroCopyOutputStream* rawOutput
);

int read_delimited_from(
  google::protobuf::io::ZeroCopyInputStream* rawInput,
  google::protobuf::MessageLite* message
);

enum Mode { READ, WRITE, NEW };

class Cursor {
 public:
  Cursor() { }
  virtual ~Cursor() { }
  virtual void SeekToFirst() = 0;
  virtual void Next() = 0;
  virtual string key() = 0;
  virtual string value() = 0;
  virtual bool valid() = 0;

  DISABLE_COPY_AND_ASSIGN(Cursor);
};

class Transaction {
 public:
  Transaction() { }
  virtual ~Transaction() { }
  virtual void Put(const string& key, const string& value) = 0;
  virtual void Commit() = 0;

  DISABLE_COPY_AND_ASSIGN(Transaction);
};

class DB {
 public:
  DB() { }
  virtual ~DB() { }
  virtual void Open(const string& source, Mode mode) = 0;
  virtual void Close() = 0;
  virtual Cursor* NewCursor() = 0;
  virtual Transaction* NewTransaction() = 0;

  DISABLE_COPY_AND_ASSIGN(DB);
};

DB* GetDB(DataParameter::DB backend);
DB* GetDB(const string& backend);

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_HPP
