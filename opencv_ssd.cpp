#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

const char* keys = "{ model  |    | Path to a model }"
                   "{ config |    | Path to a config file }"
                   "{ th     | .5 | Threshold for confidence }";

#if 0
"{ help  h     | | Print help message. }"
"{ device      |  0 | camera device number. }"
"{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }"
"{ model m     | | Path to a binary file of model contains trained weights. "
"It could be a file with extensions .caffemodel (Caffe), "
".pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet).}"
"{ config c    | | Path to a text file of model contains network configuration. "
"It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet).}"
"{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
"{ classes     | | Optional path to a text file with names of classes to label detected objects. }"
"{ mean        | | Preprocess input image by subtracting mean values. Mean values should be in BGR order and delimited by spaces. }"
"{ scale       |  1 | Preprocess input image by multiplying on a scale factor. }"
"{ width       | -1 | Preprocess input image by resizing to a specific width. }"
"{ height      | -1 | Preprocess input image by resizing to a specific height. }"
"{ rgb         |    | Indicate that model works with RGB input images instead BGR ones. }"
"{ thr         | .5 | Confidence threshold. }"
"{ nms         | .4 | Non-maximum suppression threshold. }"
"{ backend     |  0 | Choose one of computation backends: "
"0: automatically (by default), "
"1: Halide language (http://halide-lang.org/), "
"2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
"3: OpenCV implementation }"
"{ target      | 0 | Choose one of target computation devices: "
"0: CPU target (by default), "
"1: OpenCL, "
"2: OpenCL fp16 (half-float precision), "
"3: VPU }";
#endif

int main(int argc, char** argv)
{
	CommandLineParser parser(argc, argv, keys);

	chrono::system_clock::time_point  start, end;
	double msec;

	// Load a model
	start = chrono::system_clock::now();
	cout << "Loading model..." << endl;
	Net net = readNet(parser.get<String>("model"), parser.get<String>("config"));
	end = chrono::system_clock::now();
	msec = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	cout << "Time for loading: " << msec << " msec" << endl;

	// Get the name of output layer
	vector<String> layerNames = net.getLayerNames();
	vector<int> outLayers = net.getUnconnectedOutLayers();
	vector<String> outLayerName;
	outLayerName.push_back(layerNames[outLayers[0] - 1]);

	VideoCapture cap(0);
	Mat frame, blob;
	double threshold = parser.get<double>("th");
	while (1) {
		cap >> frame;

		// Create blob
		blobFromImage(frame, blob, 1.0, Size(frame.cols, frame.rows), Scalar(0, 0, 0, 0), false);
		
		// Run forward pass
		net.setInput(blob);
		vector<Mat> outputBlobs;
		vector<string> outBlobNames;
		net.forward(outputBlobs, outLayerName);

		// Draw detection results
		vector<int> classIds;
		vector<float> confidences;
		vector<Rect> boxes;
		float *data = (float*)outputBlobs[0].data;
		for (size_t i = 0; i < outputBlobs[0].total(); i += 7) {
			double conf = data[i + 2];
			if (conf > threshold) {
				int left = data[i + 3] * frame.cols;
				int top = data[i + 4] * frame.rows;
				int right = data[i + 5] * frame.cols;
				int bottom = data[i + 6] * frame.rows;
				int width = right - left + 1;
				int height = bottom - top + 1;
				classIds.push_back((int)(data[i + 1]) - 1);
				boxes.push_back(Rect(left, top, width, height));
				confidences.push_back(conf);
			}
		}
		vector<int> nmsIndex;
		NMSBoxes(boxes, confidences, threshold, 0.4, nmsIndex);
		for (size_t i = 0; i < nmsIndex.size(); ++i) {
			int idx = nmsIndex[i];
			Rect bbox = boxes[idx];
			rectangle(frame, bbox, Scalar(0, 255, 0));
			putText(frame, "0.80", Point(bbox.x, bbox.y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
		}

		// Show results
		imshow("captured", frame);
		if (waitKey(1) == 'q') {
			break;
		}
	}

	destroyAllWindows();

	return 0;
}