/*
  Claire Liu, Yu-Jing Wei
  RegionAnalyzer.hpp

  Path: include/RegionAnalyzer.hpp
  Description: Header file for RegionAnalyzer.cpp to analyze regions in images.
*/

#pragma once // Include guard

#include <opencv2/opencv.hpp>
#include <vector>

struct RegionFeatures
{

  int id = -1;

  // ---------- Basic Geometry ----------
  double area = 0.0;
  cv::Point2f centroid{0.f, 0.f};

  // ---------- Second Order Moments ----------
  double mu20 = 0.0;
  double mu02 = 0.0;
  double mu11 = 0.0;

  float theta = 0.f; // radians

  cv::Point2f e1{1.f, 0.f}; // primary axis
  cv::Point2f e2{0.f, 1.f}; // secondary axis

  float minE1 = 0.f, maxE1 = 0.f;
  float minE2 = 0.f, maxE2 = 0.f;

  cv::RotatedRect orientedBBox;
  std::vector<cv::Point> contour;
  cv::Mat mask;

  // ---------- Shape Feature Vector ----------
  double percentFilled = 0.0;
  double aspectRatio = 0.0;
  double hu[7] = {0}; // Hu invariant moments
};

class RegionAnalyzer
{
public:
  struct Params
  {
    bool keepMasks;
    int minAreaPixels;
    bool externalOnly;

    Params(bool keepMasks_ = false, int minAreaPixels_ = 20, bool externalOnly_ = true)
        : keepMasks(keepMasks_), minAreaPixels(minAreaPixels_), externalOnly(externalOnly_) {}
  };

  explicit RegionAnalyzer(const Params &p = Params()) : params_(p) {}

  bool computeFeaturesForRegion(
      const cv::Mat &labels_32s,
      int regionId,
      RegionFeatures &out) const;
  std::vector<RegionFeatures> analyzeLabels(const cv::Mat &labels_32s) const;

private:
  Params params_;

  static float primaryAxisTheta(double mu20, double mu02, double mu11);

  static void computePixelCentralMoments(
      const cv::Mat &regionMask,
      const cv::Rect &roi,
      const cv::Point2f &c,
      double &mu20, double &mu02, double &mu11);

  static void computeAxisExtentsFromMask(
      const cv::Mat &regionMask,
      const cv::Rect &roi,
      const cv::Point2f &c,
      const cv::Point2f &e1,
      const cv::Point2f &e2,
      float &minE1, float &maxE1,
      float &minE2, float &maxE2);
};

std::vector<double> getShapeFeatureVector(const RegionFeatures &r);
