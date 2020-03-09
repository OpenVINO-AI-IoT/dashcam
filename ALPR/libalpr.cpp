#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include "alpr.h"
#include "alpr_c.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/imgproc/imgproc_c.h>

#include <boost/python.hpp>
#include <boost/filesystem.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/object.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/detail/defaults_gen.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/arrayobject.h>
#include <sstream>

using namespace boost::filesystem;

using namespace alpr;
namespace bp = boost::python;
namespace np = boost::python::numpy;

#define LIB_ALPR_VERSION "0.0.1"

namespace Dashcam {

class ALPRImageDetect {

public:

    struct LicensePlate {
        std::string plateText;
        int index;
        float confidence;
        bool match;
    };

    const std::string extension = ".jpg";
    static constexpr float OVERALL_CONFIDENCE = 0.5f;

    ALPRImageDetect() {}
    ~ALPRImageDetect() {}

    void setAttributes(std::string i_config, std::string i_country, std::string i_region = "", std::string runtime_dir = "")
    {
        configFileName = i_config;
        country = i_country;
        region = i_region;

        setInstance(runtime_dir);
    }

    void setInstance(std::string runtime_dir)
    {
        instance = new alpr::Alpr(country, configFileName, runtime_dir);
    }

    void setFrame(np::ndarray src)
    {
        np::ndarray nd = np::array(src);
        rows = (int) nd.shape(0);
        cols = (int) nd.shape(1);

        char * data = nd.get_data();
        cv::Mat i_image(rows,cols,CV_8UC3,data,cv::Mat::AUTO_STEP);

        image = i_image.clone();
    }

    void setImage(char * data, int r, int c)
    {
        rows = r;
        cols = c;
        cv::Mat i_image(r, c, CV_8UC3, data, cv::Mat::AUTO_STEP);
        image = i_image.clone();
    }

    bp::object getPlacements()
    {
        return placements;
    }

    std::vector<AlprRegionOfInterest> constructRegionsOfInterest(std::vector<std::tuple<int, int, int, int>> regionsOfInterest)
    {
        int x, y, width, height;
        std::vector<AlprRegionOfInterest> regions(regionsOfInterest.size());
        for(int i = 0; i < regionsOfInterest.size(); i++) {
            std::tie(x, y, width, height) = regionsOfInterest[i];
            AlprRegionOfInterest regionOfInterest(x, y, width, height);
            regions[i] = regionOfInterest;
        }
        
        return regions;
    }

    std::vector<LicensePlate> processPlates(AlprResults results, float overall_confidence = ALPRImageDetect::OVERALL_CONFIDENCE, bool matches_template = false)
    {
        std::vector<LicensePlate> licensePlates;
        for (int i = 0; i < results.plates.size(); i++)
        {
            AlprPlateResult plate = results.plates[i];
            AlprRegionOfInterest _region = results.regionsOfInterest[i];
            for (int j = 0; j < plate.topNPlates.size(); j++)
            {
                AlprPlate candidate = plate.topNPlates[j];
                bool match = true ? (matches_template == false) : (candidate.matches_template == matches_template);
                if(candidate.overall_confidence > overall_confidence && match) {
                    licensePlates.push_back(LicensePlate{candidate.characters, i, candidate.overall_confidence, candidate.matches_template});
                }
            }
            placements.append(bp::make_tuple(
                bp::make_tuple(results.epoch_time,results.frame_number,results.img_width,results.total_processing_time_ms), 
                bp::make_tuple(_region.x,_region.y,_region.width,_region.height)));
        }

        return licensePlates;
    }

    AlprResults detectAutonomousLicensePlate(int topN, std::vector<std::tuple<int, int, int, int>> regionsOfInterest)
    {
        instance->setTopN(topN);
        cv::Size size = image.size();
        void* buffer = malloc(size.width*size.height*sizeof(int)*3);
        std::vector<uchar>* buf = reinterpret_cast<std::vector<uchar>*>(buffer);
        cv::imencode(extension, image, *buf, {cv::IMWRITE_JPEG_OPTIMIZE});
        std::vector<AlprRegionOfInterest> RegionsOfInterest = constructRegionsOfInterest(regionsOfInterest);
        std::vector<char>* charbuf = reinterpret_cast<std::vector<char>*>(buf);
        results = instance->recognize(*charbuf, RegionsOfInterest);
        return results;
    }

    std::vector<LicensePlate> detectLicensePlateMatches(int topN, std::vector<std::tuple<int, int, int, int>> regionsOfInterest, bool matches)
    {
        results = detectAutonomousLicensePlate(topN, regionsOfInterest);

        return processPlates(results, 0.5f, matches);
    }

    bp::list LicensePlate_Matches(int topN, bp::list regions_of_interest, bool matches = false)
    {
        std::vector<std::tuple<int, int, int, int>> regionsOfInterest(bp::len(regions_of_interest));
        std::vector<LicensePlate> results;

        int i = 0, a1,a2,a3,a4;
        bp::ssize_t n = bp::len(regions_of_interest);
        for(bp::ssize_t i = 0; i < n; i++) {
            bp::object elem = regions_of_interest[i];
            a1 = bp::extract<int>(elem[0]);
            a2 = bp::extract<int>(elem[1]);
            a3 = bp::extract<int>(elem[2]);
            a4 = bp::extract<int>(elem[3]);
            regionsOfInterest[i] = std::make_tuple(a1,a2,a3,a4);
        }

        results = detectLicensePlateMatches(topN, regionsOfInterest, matches);

        bp::tuple plate;
        bp::list plates;
        for(int j = 0; j < results.size(); j++) {
            plate = bp::make_tuple(results[j].plateText, results[j].confidence, results[j].match);
            plates.append(plate);
        }

        return plates;
    }

private:
    cv::Mat image;
    int rows;
    int cols;
    std::string configFileName;
    std::string country;
    std::string region;
    alpr::Alpr* instance;
    alpr::AlprResults results;
    bp::list placements;
};

}

BOOST_PYTHON_MODULE(libalpr) {

  Py_Initialize();
  np::initialize();

  using namespace boost::python;
  using namespace Dashcam;

  bp::class_<ALPRImageDetect>("ALPRImageDetect")
      .def(bp::init<>())
      .def("Attributes", &ALPRImageDetect::setAttributes)
      .def("Placements", &ALPRImageDetect::getPlacements)
      .def("SetFrame", &ALPRImageDetect::setFrame)
      .def("LicensePlate_Matches", &ALPRImageDetect::LicensePlate_Matches);

  import_array1();

}
