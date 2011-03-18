/*
 * Processor.cpp
 *
 *  Created on: Jun 13, 2010
 *      Author: ethan
 */

#include "Processor.h"

#include <sys/stat.h>

#define DEBUG 1

#if DEBUG
#include <android/log.h>
#  define  D(x...)  __android_log_print(ANDROID_LOG_INFO,"helloneon",x)
#else
#  define  D(...)  do {} while (0)
#endif

using namespace cv;

Processor::Processor() :
      stard(20/*max_size*/, 8/*response_threshold*/, 15/*line_threshold_projected*/, 8/*line_threshold_binarized*/, 5/*suppress_nonmax_size*/),
      fastd(20/*threshold*/, true/*nonmax_suppression*/),
      surfd(100./*hessian_threshold*/, 1/*octaves*/, 2/*octave_layers*/)

{

}

Processor::~Processor()
{
  // TODO Auto-generated destructor stub
}



void crossCheckMatching( Ptr<DescriptorMatcher> &descriptorMatcher,
                         const Mat& descriptors1, const Mat& descriptors2,
                         vector<DMatch>& filteredMatches12, int knn=1 )
{
    filteredMatches12.clear();
    vector<vector<DMatch> > matches12, matches21;
    descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, knn );
    descriptorMatcher->knnMatch( descriptors2, descriptors1, matches21, knn );
    for( size_t m = 0; m < matches12.size(); m++ )
    {
        bool findCrossCheck = false;
        for( size_t fk = 0; fk < matches12[m].size(); fk++ )
        {
            DMatch forward = matches12[m][fk];

            for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
            {
                DMatch backward = matches21[forward.trainIdx][bk];
                if( backward.trainIdx == forward.queryIdx )
                {
                    filteredMatches12.push_back(forward);
                    findCrossCheck = true;
                    break;
                }
            }
            if( findCrossCheck ) break;
        }
    }
}

void warpPerspectiveRand( const Mat& src, Mat& dst, Mat& H, RNG& rng )
{
    H.create(3, 3, CV_32FC1);
    H.at<float>(0,0) = rng.uniform( 0.8f, 1.2f);
    H.at<float>(0,1) = rng.uniform(-0.1f, 0.1f);
    H.at<float>(0,2) = rng.uniform(-0.1f, 0.1f)*src.cols;
    H.at<float>(1,0) = rng.uniform(-0.1f, 0.1f);
    H.at<float>(1,1) = rng.uniform( 0.8f, 1.2f);
    H.at<float>(1,2) = rng.uniform(-0.1f, 0.1f)*src.rows;
    H.at<float>(2,0) = rng.uniform( -1e-4f, 1e-4f);
    H.at<float>(2,1) = rng.uniform( -1e-4f, 1e-4f);
    H.at<float>(2,2) = rng.uniform( 0.8f, 1.2f);

    warpPerspective( src, dst, H, src.size() );
}

void validateKeypoints(const std::vector<KeyPoint>& keypoints, const vector<int>& keypointIndexes)
{
  //D("begin validateKeypoints\n");
  for( size_t i = 0; i < keypointIndexes.size(); i++ )
  {
    int idx = keypointIndexes[i];
    if( idx >= 0 ) {
      if (idx > (keypoints.size() - 1)) {
        //D("we got an out of bounds thing. idx: %d, size: %zu\n", idx, keypoints.size());
      }
    }
    else
    {
      //D("keypointIndexes has element < 0. TODO: process this case" );
      //points2f[i] = Point2f(-1, -1);
    }
  }
  //D("validateKeypoints succeeded\n");
}

void convertKeypoints(const std::vector<KeyPoint>& keypoints, std::vector<Point2f>& points2f,
    const vector<int>& keypointIndexes)
{
  if( keypointIndexes.empty() )
  {
    points2f.resize( keypoints.size() );
    for( size_t i = 0; i < keypoints.size(); i++ ) {
      D("trying to store %d, size is %zu\n", i, keypoints.size());
      points2f[i] = keypoints[i].pt;
    }
  }
  else
  {
    points2f.resize( keypointIndexes.size() );
    for( size_t i = 0; i < keypointIndexes.size(); i++ )
    {
      int idx = keypointIndexes[i];
      if( idx >= 0 ) {
        D("keypointIndexes.size(): %zu\n", keypointIndexes.size());
        D("keypoints.size(): %zu\n", keypoints.size());
        D("indexing into keypoints with: %d (aka idx)\n", idx);
        D("points2f.size(): %zu\n", points2f.size());
        D("indexing into points2f with %zu (aka i)\n", i);
        D("points2f[i] = keypoints[idx].pt;\n");
        D("points2f[%zu] = keypoints[%d].pt;\n", i, idx);
        points2f[i] = keypoints[idx].pt;
      } else {
        CV_Error( CV_StsBadArg, "keypointIndexes has element < 0. TODO: process this case" );
        //points2f[i] = Point2f(-1, -1);
      }
    }
  }
}

void doIteration( const Mat& img1, Mat& img2,
                  vector<KeyPoint>& keypoints1, const Mat& descriptors1,
                  FeatureDetector* detector, Ptr<DescriptorExtractor> &descriptorExtractor,
                  Ptr<DescriptorMatcher> &descriptorMatcher,
                  RNG& rng, Mat& drawImg )
{
    assert( !img1.empty() );
    assert( !img2.empty()/* && img2.cols==img1.cols && img2.rows==img1.rows*/ );
    Mat H12;
    D("images are valid\n");

    vector<KeyPoint> keypoints2;
    detector->detect( img2, keypoints2 );
    D("detector succeeded\n");

    Mat descriptors2;
    descriptorExtractor->compute( img2, keypoints2, descriptors2 );
    D("exatractor succeeded\n");

    vector<DMatch> filteredMatches;
    crossCheckMatching( descriptorMatcher, descriptors1, descriptors2, filteredMatches, 1 );
    D("crossCheckMatching succeeded\n");

    vector<int> *queryIdxs = new vector<int>(filteredMatches.size());
    vector<int> *trainIdxs = new vector<int>(filteredMatches.size());
    vector<Point2f> *points1 = new vector<Point2f>(filteredMatches.size());
    D("points1->size(): %d\n", points1->size());
    vector<Point2f> *points2 = new vector<Point2f>(filteredMatches.size());
    for( size_t i = 0; i < filteredMatches.size(); i++ )
    {
        queryIdxs->at(i) = filteredMatches[i].queryIdx;
        trainIdxs->at(i) = filteredMatches[i].trainIdx;
    }
    D("filteredMatches succeeded\n");

    D("queryIdxs.empty(): %d\n", queryIdxs->empty());

    validateKeypoints(keypoints1, *queryIdxs);
    convertKeypoints(keypoints1, *points1, *queryIdxs);
    D("keyPoint1::convert succeeded\n");

    D("trainIdxs.empty(): %d\n", trainIdxs->empty());

    validateKeypoints(keypoints1, *queryIdxs);
    convertKeypoints(keypoints2, *points2, *trainIdxs);
    D("keyPoint2::convert succeeded\n");

    H12 = findHomography( Mat(*points1), Mat(*points2), CV_RANSAC, 0.0 );
    D("findHomography\n");

    if( !H12.empty() ) // filter outliers
    {
        vector<char> matchesMask( filteredMatches.size(), 0 );
        vector<Point2f> points1; KeyPoint::convert(keypoints1, points1, *queryIdxs);
        vector<Point2f> points2; KeyPoint::convert(keypoints2, points2, *trainIdxs);
        Mat points1t; perspectiveTransform(Mat(points1), points1t, H12);
        D("perspectiveTransform succeeded\n");
        for( size_t i1 = 0; i1 < points1.size(); i1++ )
        {
            if( norm(points2[i1] - points1t.at<Point2f>((int)i1,0)) < 4 ) // inlier
                matchesMask[i1] = 1;
        }
        D("inliers succeeded\n");
        // draw inliers
        drawMatches( img1, keypoints1, img2, keypoints2, filteredMatches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask
                   );
        D("drawMatches (inlier)\n");
        //drawMatches( img1, keypoints1, img2, keypoints2, filteredMatches, drawImg );
        //for (vector<KeyPoint>::const_iterator it = keypoints2.begin(); it != keypoints2.end(); ++it)
        //{
        //  circle(drawImg, it->pt, 3, cvScalar(255, 0, 255, 0));
        //}
        //D("drawMatches (outlier)\n");
    }
    else
    {
        D("drawMatches (nomatches)\n");
    }
    delete points1;
    delete points2;
    D("deleting queryIdxs\n");
    delete queryIdxs;
    D("deleting trainIdxs\n");
    delete trainIdxs;
}

void Processor::setupDescriptorExtractorMatcher(const char* filename, int feature_type)
{
    descriptorExtractor = DescriptorExtractor::create("SURF");
    descriptorMatcher = DescriptorMatcher::create("FlannBased");
    D("filename: %s\n", filename);
    img1 = imread(filename);
    if (img1.empty())
    {
      D("img1 empty?: %d\n", img1.empty());
    }
    FeatureDetector* fd = 0;

    switch (feature_type)
    {
      case DETECT_SURF:
        fd = &surfd;
        break;
      case DETECT_FAST:
        fd = &fastd;
        break;
      case DETECT_STAR:
        fd = &stard;
        break;
    }

//    D("descriptorExtractor: %p, descriptorMatcher: %p, fd: %p\n", descriptorExtractor, descriptorMatcher,
//        fd);
    D("descriptorExtractor.empty(): %d, descriptorMatcher.empty(): %d, fd.empty(): %d\n",
        descriptorExtractor.empty(), descriptorMatcher.empty());//, fd.empty());
    fd->detect(img1, keypoints1);
    D("keypoints1 size: %zu\n", keypoints1.size());

    descriptorExtractor->compute(img1, keypoints1, descriptors1);
    D("compute done!\n");
    rng = theRNG();
}

void Processor::detectAndDrawFeatures(int input_idx, image_pool* pool, int feature_type)
{
  D("Processor::detectAndDrawFeatures\n");
  FeatureDetector *fd = 0;

  switch (feature_type)
  {
    case DETECT_SURF:
      fd = &surfd;
      break;
    case DETECT_FAST:
      fd = &fastd;
      break;
    case DETECT_STAR:
      fd = &stard;
      break;
  }

  Mat greyimage = pool->getGrey(input_idx);

  Mat img = pool->getImage(input_idx);

  if (img.empty() || greyimage.empty() || fd == 0)
    return; //no image at input_idx!
  D("we passed basic validation\n");

//keypoints1.clear();

  //if(grayimage->step1() > sizeof(uchar)) return;
  //cvtColor(*img,*grayimage,CV_RGB2GRAY);


    D("begin doIteration\n");
    doIteration(img1, greyimage, keypoints1, descriptors1,
                 fd, descriptorExtractor, descriptorMatcher,
                 rng, img);
    D("doIteration finished\n");

//  fd->detect(greyimage, keypoints);

//  for (vector<KeyPoint>::const_iterator it = keypoints.begin(); it != keypoints.end(); ++it)
//  {
//    circle(img, it->pt, 3, cvScalar(255, 0, 255, 0));
//  }

  //pool->addImage(output_idx,outimage);

}
static double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
                                        const vector<vector<Point2f> >& imagePoints, const vector<Mat>& rvecs,
                                        const vector<Mat>& tvecs, const Mat& cameraMatrix, const Mat& distCoeffs,
                                        vector<float>& perViewErrors)
{
  vector<Point2f> imagePoints2;
  int i, totalPoints = 0;
  double totalErr = 0, err;
  perViewErrors.resize(objectPoints.size());

  for (i = 0; i < (int)objectPoints.size(); i++)
  {
    projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
    err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L1);
    int n = (int)objectPoints[i].size();
    perViewErrors[i] = err / n;
    totalErr += err;
    totalPoints += n;
  }

  return totalErr / totalPoints;
}

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners)
{
  corners.resize(0);

  for (int i = 0; i < boardSize.height; i++)
    for (int j = 0; j < boardSize.width; j++)
      corners.push_back(Point3f(float(j * squareSize), float(i * squareSize), 0));
}

/**from opencv/samples/cpp/calibration.cpp
 *
 */
static bool runCalibration(vector<vector<Point2f> > imagePoints, Size imageSize, Size boardSize, float squareSize,
                           float aspectRatio, int flags, Mat& cameraMatrix, Mat& distCoeffs, vector<Mat>& rvecs,
                           vector<Mat>& tvecs, vector<float>& reprojErrs, double& totalAvgErr)
{
  cameraMatrix = Mat::eye(3, 3, CV_64F);
  if (flags & CV_CALIB_FIX_ASPECT_RATIO)
    cameraMatrix.at<double> (0, 0) = aspectRatio;

  distCoeffs = Mat::zeros(5, 1, CV_64F);

  vector<vector<Point3f> > objectPoints(1);
  calcChessboardCorners(boardSize, squareSize, objectPoints[0]);
  for (size_t i = 1; i < imagePoints.size(); i++)
    objectPoints.push_back(objectPoints[0]);

  calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags);

  bool ok = checkRange(cameraMatrix, CV_CHECK_QUIET) && checkRange(distCoeffs, CV_CHECK_QUIET);

  totalAvgErr
      = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

  return ok;
}

bool Processor::detectAndDrawChessboard(int idx, image_pool* pool)
{

  Mat grey = pool->getGrey(idx);
  if (grey.empty())
    return false;
  vector<Point2f> corners;

  IplImage iplgrey = grey;
  if (!cvCheckChessboard(&iplgrey, Size(6, 8)))
    return false;
  bool patternfound = findChessboardCorners(grey, Size(6, 8), corners);

  Mat img = pool->getImage(idx);

  if (corners.size() < 1)
    return false;

  cornerSubPix(grey, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

  if (patternfound)
    imagepoints.push_back(corners);

  drawChessboardCorners(img, Size(6, 8), Mat(corners), patternfound);

  imgsize = grey.size();

  return patternfound;

}

void Processor::drawText(int i, image_pool* pool, const char* ctext)
{
  // Use "y" to show that the baseLine is about
  string text = ctext;
  int fontFace = FONT_HERSHEY_COMPLEX_SMALL;
  double fontScale = .8;
  int thickness = .5;

  Mat img = pool->getImage(i);

  int baseline = 0;
  Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
  baseline += thickness;

  // center the text
  Point textOrg((img.cols - textSize.width) / 2, (img.rows - textSize.height * 2));

  // draw the box
  rectangle(img, textOrg + Point(0, baseline), textOrg + Point(textSize.width, -textSize.height), Scalar(0, 0, 255),
            CV_FILLED);
  // ... and the baseline first
  line(img, textOrg + Point(0, thickness), textOrg + Point(textSize.width, thickness), Scalar(0, 0, 255));

  // then put the text itself
  putText(img, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
}
void saveCameraParams(const string& filename, Size imageSize, Size boardSize, float squareSize, float aspectRatio,
                      int flags, const Mat& cameraMatrix, const Mat& distCoeffs, const vector<Mat>& rvecs,
                      const vector<Mat>& tvecs, const vector<float>& reprojErrs,
                      const vector<vector<Point2f> >& imagePoints, double totalAvgErr)
{
  FileStorage fs(filename, FileStorage::WRITE);

  time_t t;
  time(&t);
  struct tm *t2 = localtime(&t);
  char buf[1024];
  strftime(buf, sizeof(buf) - 1, "%c", t2);

  fs << "calibration_time" << buf;

  if (!rvecs.empty() || !reprojErrs.empty())
    fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());
  fs << "image_width" << imageSize.width;
  fs << "image_height" << imageSize.height;
  fs << "board_width" << boardSize.width;
  fs << "board_height" << boardSize.height;
  fs << "squareSize" << squareSize;

  if (flags & CV_CALIB_FIX_ASPECT_RATIO)
    fs << "aspectRatio" << aspectRatio;

  if (flags != 0)
  {
    sprintf(buf, "flags: %s%s%s%s", flags & CV_CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "", flags
        & CV_CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "", flags & CV_CALIB_FIX_PRINCIPAL_POINT
        ? "+fix_principal_point" : "", flags & CV_CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
    cvWriteComment(*fs, buf, 0);
  }

  fs << "flags" << flags;

  fs << "camera_matrix" << cameraMatrix;
  fs << "distortion_coefficients" << distCoeffs;

  fs << "avg_reprojection_error" << totalAvgErr;
  if (!reprojErrs.empty())
    fs << "per_view_reprojection_errors" << Mat(reprojErrs);

  if (!rvecs.empty() && !tvecs.empty())
  {
    Mat bigmat(rvecs.size(), 6, CV_32F);
    for (size_t i = 0; i < rvecs.size(); i++)
    {
      Mat r = bigmat(Range(i, i + 1), Range(0, 3));
      Mat t = bigmat(Range(i, i + 1), Range(3, 6));
      rvecs[i].copyTo(r);
      tvecs[i].copyTo(t);
    }
    cvWriteComment(*fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0);
    fs << "extrinsic_parameters" << bigmat;
  }

  if (!imagePoints.empty())
  {
    Mat imagePtMat(imagePoints.size(), imagePoints[0].size(), CV_32FC2);
    for (size_t i = 0; i < imagePoints.size(); i++)
    {
      Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
      Mat(imagePoints[i]).copyTo(r);
    }
    fs << "image_points" << imagePtMat;
  }
}
void Processor::resetChess()
{

  imagepoints.clear();
}

void Processor::calibrate(const char* filename)
{

  vector<Mat> rvecs, tvecs;
  vector<float> reprojErrs;
  double totalAvgErr = 0;
  int flags = 0;
  bool writeExtrinsics = true;
  bool writePoints = true;

  bool ok = runCalibration(imagepoints, imgsize, Size(6, 8), 1.f, 1.f, flags, K, distortion, rvecs, tvecs, reprojErrs,
                           totalAvgErr);

  if (ok)
  {

    saveCameraParams(filename, imgsize, Size(6, 8), 1.f, 1.f, flags, K, distortion, writeExtrinsics ? rvecs : vector<
        Mat> (), writeExtrinsics ? tvecs : vector<Mat> (), writeExtrinsics ? reprojErrs : vector<float> (), writePoints
        ? imagepoints : vector<vector<Point2f> > (), totalAvgErr);
  }

}

int Processor::getNumberDetectedChessboards()
{
  return imagepoints.size();
}
