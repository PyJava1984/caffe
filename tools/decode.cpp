#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/db_leveldb.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
namespace db = caffe::db;

int main(int argc, char** argv) {
  boost::shared_ptr<db::DB> db(db::GetDB("leveldb"));
  std::string path(argv[1]);
  db->Open(path, db::READ);
  db::Cursor *cursor = db->NewCursor();

  Datum datum;

  std::string base_folder(argv[2]);
  int num_of_images = atoi(argv[3]);
  int i = 0;

  while (cursor->valid()) {
    std::string filename = cursor->key() + ".txt";
    std::string value = cursor->value();

    datum.ParseFromString(value);

    int height = datum.height();
    int width = datum.width();
    int channels = datum.channels();
    int feature_size = datum.float_data().size();

    std::ofstream feature_file;
    feature_file.open((base_folder + "/" + filename).c_str());

    feature_file << height << " " << width << " " << channels << " " << feature_size << " ";

    if (feature_size > 0 ) {
      feature_file << datum.float_data().data()[0];

      for (int i = 1; i < feature_size; ++i) {
        feature_file << " " << datum.float_data().data()[i];
      }
    }

    feature_file.close();

    datum.clear_data();
    datum.clear_float_data();

    ++i;
    cursor->Next();
  }

  if (i != num_of_images) {
    LOG(ERROR) << "Expected " << num_of_images << " images, but only got " << i << " images.";
  }

  delete cursor;
  db->Close();
}
