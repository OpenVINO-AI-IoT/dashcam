/*
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

// ===============================================================================
// Generated file for Inference Engine extension for CPU plugin
//
// IMPLEMENT YOUR KERNEL HERE.
//
// You need to edit this file in order to:
//  1. initialize parameters (in constructor)
//  2. implement inference logic (in execute() method)
//
// Refer to the section "Inference Engine Kernels Extensibility" in 
// the OpenVINO Inference Engine Developer Guide 
// ===============================================================================

#include "ext_list.hpp"
#include "ext_base.hpp"
#include "ie_parallel.hpp"
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cstddef>
#include <omp.h>
#include <opencv2/core.hpp>

#define TOP_N 10
#define MATCHES 0
#define MAX_CHARS 22 // 20 characters for plates

using namespace Dashcam;
using namespace std;

namespace dashcam_alpr = Dashcam::ALPRImageDetect;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class alprImpl: public ExtLayerBase {
public:
    explicit alprImpl(const CNNLayer* layer) {
        try {
            // LayerSetUp
            // Read parameters from IR and/or initialise them here.
            //
            // Implemented functions for reading parameters are:
            // for single value:
            //     getParamAsFloat, getParamAsInt, getParamsAsBool, getParamAsString
            // for array
            //     getParamAsFloats, getParamAsInts
            //
            // Functions are declared in Inference Engine folder include/ie_layers.h
            //
            // Example of parameters reading is:
            //   scale_=layer->GetParamAsFloat("scale")

            
            /* Set configuration: specify data format for layer
             *   For more information about data formats see: 
             *   "Inference Engine Memory primitives" in OpenVINO documentation
             *------------------------------------------------------------------------------*/

            addConfig(layer, { DataConfigurator(ConfLayout::PLN) }, {DataConfigurator(ConfLayout::PLN) });

            std::vector<std::string> settings;
            cv::FileStorage fs("/home/dashcam/ALPR/settings.json", cv::FileStorage::READ);
            fs["settings"] >> settings;
            alprDetect = DashCam::ALPRImageDetect();
            alprDetect.setAttributes(settings[0], settings[1], settings[2], settings[3], settings[4]);

        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    std::vector<std::tuple<int, int, int, int>> getRegions(Blob::Ptr* regions, SizeVector regions_dims) {
      int BATCH = static_cast<int>((regions_dims.size() > 0) ? regions_dims[0] : 1);
      int X, Y, WIDTH, HEIGHT;
      int * set = regions->buffer().as<int*>();
      std::vector<int, int, int, int> results(BATCH);
      int COLS = 4;
      for(int i = 0; i < BATCH; i++) {
        results[i] = std::make_tuple(
          set[i*COLS+0], set[i*COLS+1], set[i*COLS+2], set[i*COLS+3]
        );
      }

      return results;
    }

    void writeOutput(float * dst_data, std::vector<dashcam_alpr::LicensePlate> plates)
    {
      std::string plateText;
      for(int i = 0; i < plates.size(); i++) {
        dst_data[i*MAX_CHARS + 0] = plates[i].confidence;
        dst_data[i*MAX_CHARS + 1] = (float) plates[i].match;
        for(size_t j = 0; j < plates[i].plateText.length(); j++) {
          dst_data[i*MAX_CHARS + 2+j] = static_cast<float>(plates[i].plateText[j]);
        }
      }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        // Add implementation for layer inference here
        // Examples of implementations are in OpenVINO samples/extensions folder
        
        // Get the dimensions from the input (output dimensions are the same)  
        SizeVector dims = inputs[0]->getTensorDesc().getDims();
        SizeVector region_dims = inputs[1]->getTensorDesc().getDims();
        float* dst_data = outputs[0]->buffer();

        // Get dimensions:N=Batch size, C=Number of Channels, H=Height, W=Width
        int N = static_cast<int>((dims.size() > 0) ? dims[0] : 1);
        int C = static_cast<int>((dims.size() > 1) ? dims[1] : 1);
        int H = static_cast<int>((dims.size() > 2) ? dims[2] : 1);
        int W = static_cast<int>((dims.size() > 3) ? dims[3] : 1);

        // Get pointers to source and destination buffers 
        char * src_data = inputs[0]->buffer().as<char*>();

        alprDetect.setImage(src_data, H, W);
        std::vector<std::tuple<int, int, int, int>> regionsOfInterest = getRegions(inputs[1], region_dims);
        std::vector<dashcam_alpr::LicensePlate> plates = 
        alprDetect.detectLicensePlateMatches(TOP_N, regionsOfInterest,  (bool) MATCHES);

        writeOutput(dst_data, plates);
        
        return OK;
    }

private:
    Dashcam::ALPRImageDetect alprDetect;
};

REG_FACTORY_FOR(ImplFactory<alprImpl>, cosh);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
