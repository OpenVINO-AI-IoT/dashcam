'''
Contains code for working with the Inference Engine.
You'll learn how to implement this code and more in
the related lesson on the topic.
'''

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

class NetworkService:
    '''
    Load and store information for working with the Inference Engine,
    and any loaded models.
    '''

    def __init__(self):
        self.plugin = None
        self.input_blob = None
        self.exec_network = None
        self.infer_requests = {}

    def load_model(self, model, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        self.model_xml = model
        self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        self.network = IENetwork(model=self.model_xml, weights=self.model_bin)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        print(self.network.outputs)

        return self.network.inputs[self.input_blob].shape

    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        return self.network.inputs[self.input_blob].shape

    def sync_inference(self, image):
        '''
        Makes a synchronous inference request, given an input image.
        '''
        self.exec_network.infer({self.input_blob: image})
        return

    def async_inference(self, image, request_id=0):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        self.infer_requests[request_id] = self.exec_network.start_async(request_id=request_id, inputs={self.input_blob: image})
        
        return self.infer_requests[request_id]

    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        statuses = []
        for request_id, request in self.infer_requests.items():
            statuses.append(request.wait(self.delay))
        
        return statuses

    def extract_output(self, request_id=0):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        return self.exec_network.requests[request_id].outputs

