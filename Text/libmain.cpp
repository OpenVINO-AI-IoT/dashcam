// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define TXT_DECODE_VERSION "0.0.1"

#include "Python.h"  // NOLINT(build/include_alpha)

#include <vector>
#include <iostream>
#include <iomanip>
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)
#include <fstream>  // NOLINT
#include <iostream>
#include <stdio.h>
#include <sstream>
#include <string>
#include <iostream>     // std::cout
#include <functional>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/object.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/detail/defaults_gen.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/text.hpp"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

using namespace std;
using namespace cv;

namespace bp = boost::python;
namespace np = boost::python::numpy;

void groups_draw(cv::Mat &src, std::vector<cv::Rect> &groups)
{
    for (int i=(int)groups.size()-1; i>=0; i--)
    {
        if (src.type() == CV_8UC3)
            cv::rectangle(src,groups.at(i).tl(),groups.at(i).br(),Scalar( 0, 255, 255 ), 3, 8 );
        else
            cv::rectangle(src,groups.at(i).tl(),groups.at(i).br(),Scalar( 255 ), 3, 8 );
    }
}

namespace Dashcam {

class TextDetection 
{
public:
  TextDetection() {}
  ~TextDetection() {}

  void Initialize(np::ndarray src, std::string classifierNM1, std::string classifier_NM2)
  {
    np::ndarray nd = np::array(src);
    rows = (int) nd.shape(0);
    cols = (int) nd.shape(1);

    char * data = nd.get_data();
    cv::Mat i_image(rows,cols,CV_8UC3,data,cv::Mat::AUTO_STEP);

    image = i_image.clone();

    ChanneliseFilters(classifierNM1, classifier_NM2);
  }

  void setImage(char * data, int r, int c)
  {
      rows = r;
      cols = c;
      cv::Mat i_image(r, c, CV_8UC3, data, cv::Mat::AUTO_STEP);
      image = i_image.clone();
  }

  cv::Ptr<cv::text::ERFilter> getFilterStage1() {
    return filterStage1;
  }

  cv::Ptr<cv::text::ERFilter> getFilterStage2() {
    return filterStage2;
  }

  vector<cv::Mat> getChannels() {
    return channels;
  }

  PyObject* Get_Image(cv::Mat i_image) {

    //2D image with 3 channels.
    npy_intp dimensions[3] = {i_image.rows, i_image.cols, i_image.channels()};

    //image.dims = 2 for a 2D image, so add another dimension for channels.
    return PyArray_SimpleNewFromData(i_image.dims + 1, (npy_intp*)&dimensions, NPY_UINT8, i_image.data);
  }

  vector<cv::Mat> createChannels() {
    vector<cv::Mat> channels;
    cv::text::computeNMChannels(image, channels, cv::text::ERFILTER_NM_IHSGrad);
    cv::Mat expr;
    size_t length = channels.size();
    // preprocess channels to include black and the degree of hue factor
    for(size_t i = 0; i < length-1; i++) {
        expr = channels[i].mul(-1);
        cv::Mat full = cv::Mat::ones(channels[i].size(), channels[i].type()) * 255;
        channels.push_back(full + expr);
    }
    return channels;
  }

  virtual cv::Ptr<cv::text::ERFilter> obtainFilterStage1(const cv::String& filename, 
        int thresholdDelta = 16, float minArea = (float)0.00015,
        float maxArea = (float)0.13, float minProbability = (float)0.2, 
        bool nonMaxSuppression = true, float minProbabilityDiff = (float)0.1) {
    cb1 = cv::text::loadClassifierNM1(filename);
    return cv::text::createERFilterNM1(cb1,thresholdDelta,minArea,maxArea,minProbability,nonMaxSuppression,minProbabilityDiff);
  }

  virtual cv::Ptr<cv::text::ERFilter> obtainFilterStage2(const cv::String& filename, float minProbability = (float)0.5) {
    cb2 = cv::text::loadClassifierNM2(filename);
    return cv::text::createERFilterNM2(cb2,minProbability);
  }

  void ChanneliseFilters(cv::String NM1_CLASSIFIER="./trained_classifierNM1.xml", 
  cv::String NM2_CLASSIFIER="./trained_classifierNM2.xml") {
    channels = createChannels();
    filterStage1 = obtainFilterStage1(NM1_CLASSIFIER);
    filterStage2 = obtainFilterStage2(NM2_CLASSIFIER);
  }

  int runFilter(vector<vector<cv::text::ERStat>> &regions) {
    for(int i = 0; i < channels.size(); i++) {
      filterStage1->run(channels[i], regions[i]);
      filterStage2->run(channels[i], regions[i]);
    }

    vector< vector<cv::Vec2i> > region_groups;
    vector<cv::Rect> groups_boxes;
    cv::text::erGrouping(image, channels, regions, region_groups, groups_boxes, cv::text::ERGROUPING_ORIENTATION_HORIZ);
    
    groups_rects.assign(groups_boxes.begin(), groups_boxes.end());

    // memory clean-up
    filterStage1.release();
    filterStage2.release();

    return 0;
  }

  void RunFilters(vector<cv::Mat> channels) {

    vector<vector<cv::text::ERStat>> regions(channels.size());
    runFilter(regions);

    regions.clear();
  }

  void Run_Filters()
  {
    vector<cv::Ptr<cv::text::ERFilter>> cb_vector { getFilterStage1(), getFilterStage2() };
    RunFilters(channels);
  }

  std::vector<std::tuple<int, int, int, int>> groupRects()
  {
    std::vector<std::tuple<int, int, int, int>> rects(groups_rects.size());
    for(int i = 0; i < groups_rects.size(); i++) {
      rects[i] = std::make_tuple(groups_rects[i].x, groups_rects[i].y, groups_rects[i].width, groups_rects[i].height);
    }
    return rects;
  }

  bp::list Groups_Rects()
  {
    bp::list rects;
    for(int i = 0; i < groups_rects.size(); i++) {
      rects.append(bp::make_tuple(groups_rects[i].x, groups_rects[i].y, groups_rects[i].width, groups_rects[i].height));
    }
    return rects;
  }

  PyObject* Groups_Draw(np::ndarray state)
  {
    np::ndarray nd = np::array(state);
    char * data = nd.get_data();
    cv::Mat state_image(rows, cols, CV_8UC3, data, cv::Mat::AUTO_STEP);

    groups_draw( state_image, groups_rects );

    return Get_Image( state_image );
  }

  private:
    cv::Mat image;
    cv::Ptr<cv::text::ERFilter> filterStage1;
    cv::Ptr<cv::text::ERFilter> filterStage2;
    cv::Ptr<cv::text::ERFilter::Callback> cb1;
    cv::Ptr<cv::text::ERFilter::Callback> cb2;
    vector<cv::Mat> channels;
    vector<cv::Rect> groups_rects;
    int rows;
    int cols;
};

}

#ifdef USE_BOOST_MODULE

BOOST_PYTHON_MODULE(libmain) {

  Py_Initialize();
  np::initialize();

  using namespace boost::python;

  bp::scope().attr("__version__") = AS_STRING(TXT_DECODE_VERSION);

  bp::class_<TextDetection>("TextDetection")
      .def("Initialize", &TextDetection::Initialize)
      .def("Run_Filters", &TextDetection::Run_Filters)
      .def("Groups_Rects", &TextDetection::Groups_Rects)
      .def("Groups_Draw", &TextDetection::Groups_Draw);

  import_array1();

}

#endif