/*
 * Copyright (c) 2016 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */

/*!
 * \file    home.cpp
 * \example home
 * \brief   Emulates opencv gaussian blur with 3x3 kernel required for text detection and image preprocessing
 */

/*
 * Inspired from examples of Radhakrishna Giduthuri
 */

////////
// Include OpenCV wrapper for image capture and display.
#include "opencv_camera_display.h"

////////
// The most important top-level OpenVX header files are "VX/vx.h" and "VX/vxu.h".
// The "VX/vx.h" includes all headers needed to support functionality of the
// OpenVX specification, except for immediate mode functions, and it includes:
//    VX/vx_types.h     -- type definitions required by the OpenVX standard
//    VX/vx_api.h       -- all framework API definitions
//    VX/vx_kernels.h   -- list of supported kernels in the OpenVX standard
//    VX/vx_nodes.h     -- easier-to-use functions for the kernels
//    VX/vx_vendors.h
// The "VX/vxu.h" defines the immediate mode utility functions (not needed here).
#include <VX/vx.h>

#define num_filters 3

////////
// Useful macros for OpenVX error checking:
//   ERROR_CHECK_STATUS     - check status is VX_SUCCESS
//   ERROR_CHECK_OBJECT     - check if the object creation is successful
#define ERROR_CHECK_STATUS( status ) { \
        vx_status status_ = (status); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
    }

#define ERROR_CHECK_OBJECT( obj ) { \
        vx_status status_ = vxGetStatus((vx_reference)(obj)); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
    }

// log_callback function implements a mechanism to print log messages
// from the OpenVX framework onto console.
void VX_CALLBACK log_callback( vx_context    context,
                   vx_reference  ref,
                   vx_status     status,
                   const vx_char string[] )
{
    printf( "LOG: [ status = %d ] %s\n", status, string );
    fflush( stdout );
}

int main( int argc, char * argv[] )
{
    const char * video_sequence = argv[1];
    const char * video_output = argv[2];
    const char * fps = argv[3];
    const char * stride = argv[4];
    CGuiModule gui( video_sequence );

    // Try grab first video frame from the sequence using cv::VideoCapture
    // and check if video frame is available
    if( !gui.Grab() )
    {
        printf( "ERROR: input has no video\n" );
        return 1;
    }

    gui.FrameWriter( video_output, (double) 25.0 );

    ////////
    // Set the application configuration parameters. Note that input video
    // sequence is an 8-bit RGB image with dimensions given by gui.GetWidth()
    // and gui.GetHeight(). The parameters for the median filter are:
    vx_uint32  width     = gui.GetWidth();
    vx_uint32  height    = gui.GetHeight();

    ////////
    // Create the OpenVX context and make sure returned context is valid and
    // register the log_callback to receive messages from OpenVX framework.
    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT( context );
    vxRegisterLogCallback( context, log_callback, vx_false_e );

    ////////
    // Create OpenVX image object for input RGB image and median filter output image.
    vx_image rgb_image       = vxCreateImage( context, width, height, VX_DF_IMAGE_RGB );
    vx_image threshold_image = vxCreateImage( context, width, height, VX_DF_IMAGE_RGB );
    ERROR_CHECK_OBJECT( rgb_image );
    ERROR_CHECK_OBJECT( threshold_image );

    vx_graph graph      = vxCreateGraph( context );
    ERROR_CHECK_OBJECT( graph );
    vx_image yuv_image  = vxCreateVirtualImage( graph, width, height, VX_DF_IMAGE_IYUV );
    vx_image gray_image = vxCreateVirtualImage( graph, width, height, VX_DF_IMAGE_U8 );
    ERROR_CHECK_OBJECT( yuv_image );
    ERROR_CHECK_OBJECT( gray_image );

    vx_image filters[num_filters];

    for (int i = 0; i < num_filters; ++i) {
        filters[i] = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
    }

    vxColorConvertNode( graph, rgb_image, yuv_image );
    vxChannelExtractNode(graph, yuv_image, VX_CHANNEL_Y, gray_image);
    vxChannelExtractNode(graph, rgb_image, VX_CHANNEL_R, filters[0]);
    vxChannelExtractNode(graph, rgb_image, VX_CHANNEL_G, filters[1]);
    vxChannelExtractNode(graph, rgb_image, VX_CHANNEL_B, filters[2]);
    /* Now, run Gaussian filter on each of gray scale images */
    for(int i = 5; i < num_filters + 2; i++) {
        vxGaussian3x3Node(graph, filters[i - 5], filters[i - 2]);
    }
    
    /* Now combine images together in the ouptut color image */
    vxChannelCombineNode(graph, gray_image, filters[num_filters - 3], 
    filters[num_filters - 2], filters[num_filters - 1], threshold_image);

    ERROR_CHECK_STATUS( vxReleaseImage( &yuv_image ) );
    for (int i = 0; i < num_filters; ++i) {
        vxReleaseImage(&filters[i]);
    }
    ERROR_CHECK_STATUS( vxVerifyGraph( graph ) );

    int st = std::atoi(stride);

    ////////
    // Process the video sequence frame by frame until the end of sequence or aborted.
    for( int frame_index = 0; !gui.AbortRequested(); frame_index++ )
    {
        ////////
        // Copy input RGB frame from OpenCV to OpenVX. In order to do this,
        // you need to use vxCopyImagePatch API.
        // See "VX/vx_api.h" for the description of these APIs.
        vx_rectangle_t cv_rgb_image_region;
        cv_rgb_image_region.start_x    = 0;
        cv_rgb_image_region.start_y    = 0;
        cv_rgb_image_region.end_x      = width;
        cv_rgb_image_region.end_y      = height;
        vx_imagepatch_addressing_t cv_rgb_image_layout;
        cv_rgb_image_layout.stride_x   = st;
        cv_rgb_image_layout.stride_y   = gui.GetStride();
        vx_uint8 * cv_rgb_image_buffer = gui.GetBuffer();
        ERROR_CHECK_STATUS( vxCopyImagePatch( rgb_image, &cv_rgb_image_region, 0,
                                              &cv_rgb_image_layout, cv_rgb_image_buffer,
                                              VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST ) );

        ////////
        // Now that input RGB image is ready, just run the graph.
        ERROR_CHECK_STATUS( vxProcessGraph( graph ) );

        ////////
        // Display the output filtered image.
        vx_rectangle_t rect = { 0, 0, width, height };
        vx_map_id map_id;
        vx_imagepatch_addressing_t addr;
        void * ptr;
        ERROR_CHECK_STATUS( vxMapImagePatch( threshold_image, &rect, 0, &map_id, &addr, &ptr,
                                             VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X ) );
        cv::Mat mat( height, width, CV_8U, ptr, addr.stride_y );
#if ENABLE_DISPLAY
        cv::imshow( "Gaussian 3x3 Blur", mat );
#endif
        ERROR_CHECK_STATUS( vxUnmapImagePatch( threshold_image, map_id ) );

        ////////
        // Display the results and grab the next input RGB frame for the next iteration.
        // gui.Show();
        if( !gui.Grab() )
        {
            // Terminate the processing loop if the end of sequence is detected.
            // gui.WaitForKey();
            break;
        }

        gui.Write( mat );
    }

    gui.Release();

    ////////********
    // Query graph performance using VX_GRAPH_PERFORMANCE and print timing
    // in milliseconds. Note that time units of vx_perf_t fields are nanoseconds.
    vx_perf_t perf = { 0 };
    ERROR_CHECK_STATUS( vxQueryGraph( graph, VX_GRAPH_PERFORMANCE, &perf, sizeof( perf ) ) );
    printf( "GraphName NumFrames Avg(ms) Min(ms)\n"
            "Gaussian    %9d %7.3f %7.3f\n",
            ( int )perf.num, ( float )perf.avg * 1e-6f, ( float )perf.min * 1e-6f );

    ////////********
    // To release an OpenVX object, you need to call vxRelease<Object> API which takes a pointer to the object.
    // If the release operation is successful, the OpenVX framework will reset the object to NULL.
    ERROR_CHECK_STATUS( vxReleaseGraph( &graph ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &rgb_image ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &gray_image ) );
    ERROR_CHECK_STATUS( vxReleaseContext( &context ) );

    return 0;
}
