#ifndef ETROBO_OBJECT_DETECTION__OBJECT_DETECTION_TFLITE_HPP_
#define ETROBO_OBJECT_DETECTION__OBJECT_DETECTION_TFLITE_HPP_

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

namespace etrobo_object_detection_tflite {

struct Detection {
  cv::Rect bbox;
  int class_id;
  float score;
};

class ObjectDetectionTFliteNode : public rclcpp::Node {
public:
  ObjectDetectionTFliteNode();

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  cv::Mat preprocess(const cv::Mat &img_bgr, cv::Size &orig_size);
  std::vector<Detection> postprocess(const cv::Size &orig_size,
                                     float conf_thres = 0.5f);
  void load_labels(const std::string &label_path);

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::vector<std::string> labels_;

  static constexpr int INPUT_SIZE = 300;
  static constexpr const char *MODEL_PATH = "detect.tflite";
  static constexpr const char *LABEL_PATH = "labelmap.txt";

  int input_tensor_index_;
  std::vector<int> output_tensor_indices_;
};

} // namespace etrobo_object_detection_tflite

#endif // ETROBO_OBJECT_DETECTION__OBJECT_DETECTION_TFLITE_HPP_
