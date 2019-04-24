#include <iostream>
#include <tuple>
using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
#include "darknet.h"

#define POSE_MAX_PEOPLE 96
#define NET_OUT_CHANNELS 57 // 38 for pafs, 19 for parts


static network* net;
extern "C" {
  void run_face_cv(int, char**);
  image mat_to_image(Mat);
  Mat image_to_mat(image);
}

static void draw_detections(Mat im, detection* dets, int num, float thresh)
{
  for (int i=0; i<num; i++) {
    if (dets[i].prob[0] > thresh) {
      printf("Face: %f%%\n", dets[i].prob[0]*100.0);
      box b = dets[i].bbox;
      int left = (b.x - (b.w/2.0f)) * im.cols;
      int top = (b.y- (b.h/2.0f)) * im.rows;
      int width = b.w * im.cols;
      int height = b.h * im.rows;
      Rect roi(left, top, width, height);
      Mat img_roi = im(roi);
      Mat blur_img = img_roi.clone();
      Mat mask = Mat::zeros(img_roi.rows, img_roi.cols, CV_8UC1);
      ellipse(mask,
              Point(width/2, height/2),
              Size(width/2, height/2),
              0.0f,
              0.0f,
              360.0f,
              Scalar(255, 255, 255),
              -1
              );
      //blur(img_roi, blur_img, Size(30, 30));
      GaussianBlur(img_roi, blur_img, Size(41, 41), 0);
      blur_img.copyTo(img_roi, mask);
    }
  }
}

void run_face_cv(int ac, char **av)
{
    if (ac < 5) {
        cout << "usage: ./darknet face [detect/demo] <cfg file> <weight file> <img file>" << endl;
        return;
    }

    // 1. read args
    VideoCapture* cap = NULL;
    char *cfg_path = av[3];
    char *weight_path = av[4];
    char *im_path = av[5];
    int demo_done = 0;
    Mat im;
    if (strcmp(av[2], "demo") == 0) {
      printf("video file: %s\n", im_path);
      cap = (VideoCapture*)open_video_stream(im_path, 0, 0, 0, 0);
    }
    if (cap) {
      *cap >> im;
      make_window("Demo", im.size().width, im.size().height, 0);
    }
    else {
      demo_done = 1;
      im = imread(im_path);
      if (im.empty()) {
        cout << "failed to read image" << endl;
        return;
      }
    }

    // 2. initialize net
    net = load_network(cfg_path, weight_path, 0);
    set_batch_network(net, 1);

    int net_inw = net->w;
    int net_inh = net->h;

    do {
      image img = mat_to_image(im);
      image sized = letterbox_image(img, net_inw, net_inh);

      double time_begin = getTickCount();
      network_predict(net, sized.data);

      double fee_time = (getTickCount() - time_begin) / getTickFrequency() * 1000;
      cout << "forward fee: " << fee_time << "ms" << endl;

      int nboxes = 0;
      detection* dets = get_network_boxes(net, im.cols, im.rows, 0.5, 0.5, 0, 1, &nboxes);
      do_nms_sort(dets, nboxes, net->layers[net->n-1].classes, 0.45);
      draw_detections(im, dets, nboxes, 0.5);

      free_detections(dets, nboxes);
      imshow("Demo", im);
      imwrite("demo.jpg", im);

      if (cap) {
        *cap >> im;
        if (im.empty()) {
          demo_done = 1;
        }
      }
      waitKey(demo_done? 0: 1);
      if (demo_done) {
        cout << "face: " << nboxes << endl;
      }
      free_image(img);
      free_image(sized);
    } while (!demo_done);

    return;
}

