"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def draw_boxes(frame, result, hist):
    count_now = 0
    
    for box in result[0][0]:
        if box[2] > prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            count_now = count_now + 1
            
    if count_now > 0:
        hist = 35
    elif (count_now == 0) and (hist > 0):
        count_now = 1
        hist += -1

    return frame, count_now, hist
   

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    mode_single_image = False
    request_id_now = 0
    starting_time = 0
    count_end = 0
    count_total = 0
    
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1, request_id_now, args.cpu_extension)[1]
    ### TODO: Handle the input stream ###
    if args.input == 'CAM':
        input_stream = 0
    elif args.input.endswith('.bmp') or args.input.endswith('.jpg') :
        mode_single_image = True
        input_stream = args.input
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "The input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)
    
    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR! video source cannot be open")
    global width
    global height
    global prob_threshold
    global hist
   
    width = cap.get(3)
    height = cap.get(4)
    prob_threshold = args.prob_threshold
    hist = -1
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        ### TODO: Start asynchronous inference for specified request ###
        inf_start = time.time()
        infer_network.exec_net(request_id_now, image)
        ### TODO: Wait for the result ###
        if infer_network.wait(request_id_now) == 0:
            inf_time = time.time() - inf_start
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(request_id_now)
            ### TODO: Extract any desired stats from the results ###
            frame, count_now, hist = draw_boxes(frame, result, hist)
            inference_time_message = "Time for Inference: {:.2f}ms"\
                               .format(inf_time * 1000)
            cv2.putText(frame, inference_time_message, (40, 400),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.9, (102, 51, 0), 2)
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if count_now > count_end:
                starting_time = time.time()
                count_total = count_total + count_now - count_end
                client.publish("person", json.dumps({"total": count_total}))

            if count_now < count_end and int(time.time() - starting_time) >2:
                time_in_frame = int(time.time() - starting_time)
                client.publish("person/duration",
                               json.dumps({"duration": time_in_frame}))

            client.publish("person", json.dumps({"count": count_now}))
            count_end = count_now

            if key_pressed == 27:
                break

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if mode_single_image:
            cv2.imwrite('output_image.jpg', frame)
            
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    # Perform inference on the input stream
    infer_on_stream(args, client)
    draw_boxes(frame, result)


if __name__ == '__main__':
    main()

