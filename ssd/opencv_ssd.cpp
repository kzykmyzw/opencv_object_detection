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
			string label = format("%.2f", conf);
			if (!classes.empty())
			{
				CV_Assert(classId < (int)classes.size());
				label = classes[classId] + ": " + label;
			}

			int baseLine;
			Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

			top = max(top, labelSize.height);
			rectangle(frame, Point(left, top - labelSize.height),
				Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
			putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
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