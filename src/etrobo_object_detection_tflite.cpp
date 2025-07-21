#include "etrobo_object_detection_tflite/etrobo_object_detection_tflite.hpp"
#include <fstream>
#include <iostream>

namespace etrobo_object_detection_tflite {

ObjectDetectionTFliteNode::ObjectDetectionTFliteNode()
    : Node("tflite_int8_detector") {
  model_ = tflite::FlatBufferModel::BuildFromFile(MODEL_PATH);
  if (!model_) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load model from %s",
                 MODEL_PATH);
    return;
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);

  if (!interpreter_) {
    RCLCPP_ERROR(this->get_logger(), "Failed to create interpreter");
    return;
  }

  interpreter_->SetNumThreads(2);

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    RCLCPP_ERROR(this->get_logger(), "Failed to allocate tensors");
    return;
  }

  input_tensor_index_ = interpreter_->inputs()[0];

  for (int i = 0; i < interpreter_->outputs().size(); ++i) {
    output_tensor_indices_.push_back(interpreter_->outputs()[i]);
  }

  load_labels(LABEL_PATH);

  subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      // "/camera_top/camera_top/image_raw", 1,
      "/image_raw", 1,
      std::bind(&ObjectDetectionTFliteNode::image_callback, this,
                std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(),
              "TensorFlow Lite object detection node initialized");
}

void ObjectDetectionTFliteNode::image_callback(
    const sensor_msgs::msg::Image::SharedPtr msg) {
  auto start_time = std::chrono::steady_clock::now();

  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception &e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  cv::Size orig_size;
  cv::Mat processed_img = preprocess(cv_ptr->image, orig_size);

  TfLiteTensor *input_tensor = interpreter_->tensor(input_tensor_index_);
  std::memcpy(input_tensor->data.uint8, processed_img.data,
              INPUT_SIZE * INPUT_SIZE * 3 * sizeof(uint8_t));

  if (interpreter_->Invoke() != kTfLiteOk) {
    RCLCPP_ERROR(this->get_logger(), "Failed to invoke interpreter");
    return;
  }

  std::vector<Detection> detections = postprocess(orig_size);

  for (const auto &det : detections) {
    cv::rectangle(cv_ptr->image, det.bbox, cv::Scalar(0, 255, 0), 2);

    std::string label =
        (det.class_id < labels_.size()) ? labels_[det.class_id] : "unknown";
    std::string text = label + " " + std::to_string(det.score).substr(0, 4);

    cv::putText(cv_ptr->image, text, cv::Point(det.bbox.x, det.bbox.y - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
  }

  cv::imshow("TFLite-INT8 SSD", cv_ptr->image);
  cv::waitKey(1);

  auto end_time = std::chrono::steady_clock::now();
  auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                       "latency %.1f ms", float(latency.count()) * 1e-3);
}

cv::Mat ObjectDetectionTFliteNode::preprocess(const cv::Mat &img_bgr,
                                              cv::Size &orig_size) {
  orig_size = img_bgr.size();

  cv::Mat img_rgb;
  cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

  cv::Mat img_resized;
  cv::resize(img_rgb, img_resized, cv::Size(INPUT_SIZE, INPUT_SIZE));

  return img_resized;
}

std::vector<Detection>
ObjectDetectionTFliteNode::postprocess(const cv::Size &orig_size,
                                       float conf_thres) {
  std::vector<Detection> detections;

  TfLiteTensor *boxes_tensor = interpreter_->tensor(output_tensor_indices_[0]);
  TfLiteTensor *classes_tensor =
      interpreter_->tensor(output_tensor_indices_[1]);
  TfLiteTensor *scores_tensor = interpreter_->tensor(output_tensor_indices_[2]);
  TfLiteTensor *count_tensor = interpreter_->tensor(output_tensor_indices_[3]);

  float *boxes = boxes_tensor->data.f;
  float *classes = classes_tensor->data.f;
  float *scores = scores_tensor->data.f;
  float *count = count_tensor->data.f;

  int num_detections = static_cast<int>(*count);

  for (int i = 0; i < num_detections; ++i) {
    float score = scores[i];
    if (score < conf_thres)
      continue;

    float y1 = boxes[i * 4 + 0];
    float x1 = boxes[i * 4 + 1];
    float y2 = boxes[i * 4 + 2];
    float x2 = boxes[i * 4 + 3];

    int x1_abs = static_cast<int>(x1 * orig_size.width);
    int y1_abs = static_cast<int>(y1 * orig_size.height);
    int x2_abs = static_cast<int>(x2 * orig_size.width);
    int y2_abs = static_cast<int>(y2 * orig_size.height);

    Detection det;
    det.bbox = cv::Rect(x1_abs, y1_abs, x2_abs - x1_abs, y2_abs - y1_abs);
    det.class_id = static_cast<int>(classes[i]);
    det.score = score;

    detections.push_back(det);
  }

  return detections;
}

void ObjectDetectionTFliteNode::load_labels(const std::string &label_path) {
  std::ifstream file(label_path);
  if (!file.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to open label file: %s",
                 label_path.c_str());
    return;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    labels_.push_back(line);
  }

  RCLCPP_INFO(this->get_logger(), "Loaded %zu labels", labels_.size());
}

} // namespace etrobo_object_detection_tflite

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<
      etrobo_object_detection_tflite::ObjectDetectionTFliteNode>();

  try {
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(node->get_logger(), "Exception: %s", e.what());
  }

  rclcpp::shutdown();
  return 0;
}
