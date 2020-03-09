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

#define MAX_CHARS 4

using namespace Dashcam;
using namespace std;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class textImpl: public ExtLayerBase {
public:
    explicit textImpl(const CNNLayer* layer) {
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

            // Text detection Configuration Values
            std::vector<std::string> settings;
            cv::FileStorage fs("/home/dashcam/Text/settings.json", cv::FileStorage::READ);
            fs["settings"] >> settings;

            textDetect = DashCam::TextDetection();
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    void writeOutput(float * dst_data, std::vector<std::tuple<int, int, int, int>> groups_rects)
    {
        for(int i = 0; i < groups_rects.size(); i++) {
            dst_data[i*MAX_CHARS + 0] = (float) groups_rects[i][0];
            dst_data[i*MAX_CHARS + 1] = (float) groups_rects[i][1];
            dst_data[i*MAX_CHARS + 2] = (float) groups_rects[i][2];
            dst_data[i*MAX_CHARS + 3] = (float) groups_rects[i][3];
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        // Add implementation for layer inference here
        // Examples of implementations are in OpenVINO samples/extensions folder
        
        // Get the dimensions from the input (output dimensions are the same)  
        SizeVector dims = inputs[0]->getTensorDesc().getDims();
        SizeVector regions = inputs[1]->getTensorDesc().getDims();
        float* dst_data = outputs[0]->buffer();

        // Get dimensions:N=Batch size, C=Number of Channels, H=Height, W=Width
        int N = static_cast<int>((dims.size() > 0) ? dims[0] : 1);
        int C = static_cast<int>((dims.size() > 1) ? dims[1] : 1);
        int H = static_cast<int>((dims.size() > 2) ? dims[2] : 1);
        int W = static_cast<int>((dims.size() > 3) ? dims[3] : 1);

        // Get pointers to source and destination buffers 
        char * src_data = inputs[0]->buffer().as<char*>();

        textDetect.setImage(src_data, H, W);
        textdetect.Run_Filters();
        std::vector<std::tuple<int, int, int, int>> groups_rects = textDetect.groupRects();

        writeOutput(dst_data, groups_rects);
        
        return OK;
    }

private:
  DashCam::TextDetection textDetect;
};

REG_FACTORY_FOR(ImplFactory<textImpl>, cosh);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine