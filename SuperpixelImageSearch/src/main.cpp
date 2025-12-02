#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <atomic>
#include <limits>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include "json.hpp" // nlohmann::json

namespace fs = std::filesystem;
using json = nlohmann::json;

// ---------- CONFIG: paths are relative to build/ ----------
const std::string INDEX_DIR    = "../data/coco2017/images/train2017";
const std::string QUERY_IMG    = "../data/coco2017/images/val2017/000000000139.jpg";
const std::string TRAIN_ANN    = "../data/coco2017/annotations/instances_train2017.json";
const std::string VAL_ANN      = "../data/coco2017/annotations/instances_val2017.json";
// ---------------------------------------------------------

enum class FeatureType    { SIFT, ORB };
enum class DescriptorMode { GLOBAL, SUPERPIXEL };

// ---------- COCO label index ----------
struct COCOLabelIndex
{
    // image file_name -> set of category IDs
    std::unordered_map<std::string, std::unordered_set<int>> imageToCats;
    // category_id -> category name
    std::unordered_map<int, std::string> catIdToName;
};

// Load a COCO annotation file and merge into index
void loadCOCOAnnotations(const std::string& annPath, COCOLabelIndex& index)
{
    std::ifstream f(annPath);
    if (!f.is_open())
    {
        std::cerr << "Could not open COCO annotation file: " << annPath << "\n";
        return;
    }

    json j;
    f >> j;

    // categories: fill id -> name
    if (j.contains("categories") && j["categories"].is_array())
    {
        for (const auto& cat : j["categories"])
        {
            int id = cat.value("id", -1);
            std::string name = cat.value("name", "");
            if (id >= 0 && !name.empty())
                index.catIdToName[id] = name;
        }
    }

    // images: image_id -> file_name
    std::unordered_map<int, std::string> imageIdToFile;
    if (j.contains("images") && j["images"].is_array())
    {
        for (const auto& img : j["images"])
        {
            int id = img.value("id", -1);
            std::string fname = img.value("file_name", "");
            if (id >= 0 && !fname.empty())
                imageIdToFile[id] = fname;
        }
    }

    // annotations: image_id + category_id
    if (j.contains("annotations") && j["annotations"].is_array())
    {
        for (const auto& ann : j["annotations"])
        {
            int imgId = ann.value("image_id", -1);
            int catId = ann.value("category_id", -1);
            if (imgId < 0 || catId < 0) continue;

            auto it = imageIdToFile.find(imgId);
            if (it == imageIdToFile.end()) continue;

            const std::string& fname = it->second;
            index.imageToCats[fname].insert(catId);
        }
    }

    std::cout << "Loaded COCO annotations from: " << annPath << "\n";
}

// Get category IDs for a given image filename (basename only)
std::vector<int> getCategoriesForImage(const COCOLabelIndex& index,
                                       const std::string& fullPath)
{
    std::string fname = fs::path(fullPath).filename().string();
    auto it = index.imageToCats.find(fname);
    if (it == index.imageToCats.end())
        return {};

    std::vector<int> out(it->second.begin(), it->second.end());
    std::sort(out.begin(), out.end());
    return out;
}

// Convert category IDs to a string like "toilet|sink|chair"
std::string catIdsToString(const std::vector<int>& ids,
                           const COCOLabelIndex& index)
{
    std::vector<std::string> names;
    names.reserve(ids.size());
    for (int id : ids)
    {
        auto it = index.catIdToName.find(id);
        if (it != index.catIdToName.end())
            names.push_back(it->second);
        else
            names.push_back("id_" + std::to_string(id));
    }
    std::sort(names.begin(), names.end());
    std::string out;
    for (size_t i = 0; i < names.size(); ++i)
    {
        if (i > 0) out += "|";
        out += names[i];
    }
    return out;
}

// ---------- Utility: grid-based "superpixels" ----------
void makeGridSuperpixels(const cv::Mat& bgr,
                         cv::Mat& labels,
                         int& numSuperpixels,
                         int cellSize = 32)
{
    CV_Assert(bgr.type() == CV_8UC3);
    const int h = bgr.rows;
    const int w = bgr.cols;

    const int gridX = (w + cellSize - 1) / cellSize;
    const int gridY = (h + cellSize - 1) / cellSize;

    labels.create(h, w, CV_32S);

    for (int y = 0; y < h; ++y)
    {
        int gy = y / cellSize;
        for (int x = 0; x < w; ++x)
        {
            int gx = x / cellSize;
            int sp = gy * gridX + gx;
            labels.at<int>(y, x) = sp;
        }
    }

    numSuperpixels = gridX * gridY;
}

// ---------- Feature extraction ----------
void computeFeatures(const cv::Mat& gray,
                     FeatureType type,
                     std::vector<cv::KeyPoint>& keypoints,
                     cv::Mat& descriptors,
                     int& descDim)
{
    if (type == FeatureType::SIFT)
    {
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
        sift->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
        descDim = 128;

        if (descriptors.empty())
        {
            descriptors = cv::Mat(0, descDim, CV_32F);
        }
        else if (descriptors.type() != CV_32F)
        {
            descriptors.convertTo(descriptors, CV_32F);
        }
    }
    else // ORB
    {
        cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
        orb->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
        descDim = 32;

        if (descriptors.empty())
        {
            descriptors = cv::Mat(0, descDim, CV_32F);
        }
        else
        {
            descriptors.convertTo(descriptors, CV_32F);
        }
    }
}

cv::Mat globalDescriptorMean(const cv::Mat& desc, int descDim)
{
    if (desc.empty())
        return cv::Mat::zeros(1, descDim, CV_32F);

    cv::Mat mean;
    cv::reduce(desc, mean, 0, cv::REDUCE_AVG, CV_32F);
    return mean;
}

void computeSuperpixelLABStats(const cv::Mat& bgr,
                               const cv::Mat& labels,
                               cv::Mat& globalMeanLab,
                               cv::Mat& meanLabPerSp)
{
    CV_Assert(bgr.type() == CV_8UC3);
    CV_Assert(labels.type() == CV_32S);
    CV_Assert(bgr.rows == labels.rows && bgr.cols == labels.cols);

    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);

    int h = bgr.rows;
    int w = bgr.cols;

    double minVal, maxVal;
    cv::minMaxLoc(labels, &minVal, &maxVal);
    int numSp = static_cast<int>(maxVal) + 1;

    meanLabPerSp = cv::Mat::zeros(numSp, 3, CV_32F);
    std::vector<int> counts(numSp, 0);

    for (int y = 0; y < h; ++y)
    {
        const int* lblRow = labels.ptr<int>(y);
        const cv::Vec3b* labRow = lab.ptr<cv::Vec3b>(y);
        for (int x = 0; x < w; ++x)
        {
            int sp = lblRow[x];
            const cv::Vec3b& pix = labRow[x];
            meanLabPerSp.at<float>(sp, 0) += static_cast<float>(pix[0]);
            meanLabPerSp.at<float>(sp, 1) += static_cast<float>(pix[1]);
            meanLabPerSp.at<float>(sp, 2) += static_cast<float>(pix[2]);
            counts[sp]++;
        }
    }

    for (int i = 0; i < numSp; ++i)
    {
        if (counts[i] > 0)
        {
            meanLabPerSp.at<float>(i, 0) /= counts[i];
            meanLabPerSp.at<float>(i, 1) /= counts[i];
            meanLabPerSp.at<float>(i, 2) /= counts[i];
        }
    }

    cv::reduce(meanLabPerSp, globalMeanLab, 0, cv::REDUCE_AVG, CV_32F);
}

std::vector<std::vector<int>> assignKeypointsToSuperpixels(
        const std::vector<cv::KeyPoint>& keypoints,
        const cv::Mat& labels)
{
    CV_Assert(labels.type() == CV_32S);
    int h = labels.rows;
    int w = labels.cols;

    double minVal, maxVal;
    cv::minMaxLoc(labels, &minVal, &maxVal);
    int numSp = static_cast<int>(maxVal) + 1;

    std::vector<std::vector<int>> spToIndices(numSp);

    for (int i = 0; i < (int)keypoints.size(); ++i)
    {
        float xf = keypoints[i].pt.x;
        float yf = keypoints[i].pt.y;
        int x = static_cast<int>(std::round(xf));
        int y = static_cast<int>(std::round(yf));
        if (x >= 0 && x < w && y >= 0 && y < h)
        {
            int sp = labels.at<int>(y, x);
            spToIndices[sp].push_back(i);
        }
    }

    return spToIndices;
}

// ---------- GLOBAL descriptor ----------
cv::Mat buildGlobalDescriptor(const cv::Mat& bgr, FeatureType type)
{
    CV_Assert(bgr.type() == CV_8UC3);

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat desc;
    int descDim = 0;
    computeFeatures(gray, type, keypoints, desc, descDim);

    cv::Mat globalFeat = globalDescriptorMean(desc, descDim);

    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    cv::Scalar labMeanScalar = cv::mean(lab);
    cv::Mat labMean(1, 3, CV_32F);
    labMean.at<float>(0, 0) = static_cast<float>(labMeanScalar[0]);
    labMean.at<float>(0, 1) = static_cast<float>(labMeanScalar[1]);
    labMean.at<float>(0, 2) = static_cast<float>(labMeanScalar[2]);

    cv::Mat descriptor(1, descDim + 3, CV_32F);
    globalFeat.copyTo(descriptor.colRange(0, descDim));
    labMean.copyTo(descriptor.colRange(descDim, descDim + 3));

    cv::normalize(descriptor, descriptor);
    return descriptor;
}

// ---------- SUPERPIXEL descriptor ----------
cv::Mat buildSuperpixelDescriptor(const cv::Mat& bgr, FeatureType type)
{
    CV_Assert(bgr.type() == CV_8UC3);

    cv::Mat labels;
    int numSp = 0;
    makeGridSuperpixels(bgr, labels, numSp, 32);

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat desc;
    int descDim = 0;
    computeFeatures(gray, type, keypoints, desc, descDim);

    cv::Mat globalFeat = globalDescriptorMean(desc, descDim);

    cv::Mat globalMeanLab, meanLabPerSp;
    computeSuperpixelLABStats(bgr, labels, globalMeanLab, meanLabPerSp);

    auto spToIdx = assignKeypointsToSuperpixels(keypoints, labels);
    cv::Mat regionMeans = cv::Mat::zeros(numSp, descDim, CV_32F);

    for (int sp = 0; sp < numSp; ++sp)
    {
        const auto& idxs = spToIdx[sp];
        if (idxs.empty() || desc.empty())
            continue;

        cv::Mat sum = cv::Mat::zeros(1, descDim, CV_32F);
        for (int idx : idxs)
            sum += desc.row(idx);
        sum /= static_cast<float>(idxs.size());
        sum.copyTo(regionMeans.row(sp));
    }

    cv::Mat globalRegionFeat;
    cv::reduce(regionMeans, globalRegionFeat, 0, cv::REDUCE_AVG, CV_32F);

    int totalDim = descDim + 3 + descDim;
    cv::Mat descriptor(1, totalDim, CV_32F);
    globalFeat.copyTo(descriptor.colRange(0, descDim));
    globalMeanLab.copyTo(descriptor.colRange(descDim, descDim + 3));
    globalRegionFeat.copyTo(descriptor.colRange(descDim + 3, totalDim));

    cv::normalize(descriptor, descriptor);
    return descriptor;
}

cv::Mat buildDescriptor(const cv::Mat& bgr,
                        FeatureType type,
                        DescriptorMode mode)
{
    if (mode == DescriptorMode::GLOBAL)
        return buildGlobalDescriptor(bgr, type);
    else
        return buildSuperpixelDescriptor(bgr, type);
}

// ---------- In-memory index ----------
struct ImageIndex
{
    std::vector<std::string> filenames;
    cv::Mat features;

    void add(const std::string& fname, const cv::Mat& desc)
    {
        if (features.empty())
        {
            features = desc.clone();
        }
        else
        {
            cv::Mat tmp;
            cv::vconcat(features, desc, tmp);
            features = std::move(tmp);
        }
        filenames.push_back(fname);
    }

    std::vector<std::pair<int, float>> search(const cv::Mat& query, int k) const
    {
        std::vector<std::pair<int, float>> results;
        if (features.empty())
            return results;

        CV_Assert(query.rows == 1);

        cv::Mat qRepeat;
        cv::repeat(query, features.rows, 1, qRepeat);

        cv::Mat diff, dists;
        cv::pow(features - qRepeat, 2, diff);
        cv::reduce(diff, dists, 1, cv::REDUCE_SUM);
        cv::sqrt(dists, dists);

        results.reserve(features.rows);
        for (int i = 0; i < features.rows; ++i)
        {
            float d = dists.at<float>(i, 0);
            results.emplace_back(i, d);
        }

        std::partial_sort(results.begin(),
                          results.begin() + std::min(k, (int)results.size()),
                          results.end(),
                          [](auto& a, auto& b){ return a.second < b.second; });

        if ((int)results.size() > k)
            results.resize(k);

        return results;
    }
};

// ---------- Misc helpers ----------
bool isImageFile(const fs::path& p)
{
    if (!p.has_extension()) return false;
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

struct ImageJobResult
{
    std::string path;
    cv::Mat desc;
    bool ok = false;
};

ImageJobResult processImageJob(const std::string& path,
                               FeatureType type,
                               DescriptorMode mode)
{
    ImageJobResult result;
    result.path = path;

    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "Could not read " << path << std::endl;
        result.ok = false;
        return result;
    }

    try
    {
        result.desc = buildDescriptor(img, type, mode);
        result.ok = true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error processing " << path << ": " << e.what() << std::endl;
        result.ok = false;
    }

    return result;
}

// ---------- main ----------
int main(int argc, char** argv)
{
    try
    {
        // ---- Parse CLI arguments ----
        FeatureType featureType = FeatureType::SIFT;
        std::string featureName = "SIFT";

        DescriptorMode mode = DescriptorMode::GLOBAL;
        std::string modeName = "GLOBAL";

        size_t maxImages = 1000;

        if (argc >= 2)
        {
            std::string arg1 = argv[1];
            std::transform(arg1.begin(), arg1.end(), arg1.begin(), ::tolower);
            if (arg1 == "orb")  { featureType = FeatureType::ORB;  featureName = "ORB"; }
            if (arg1 == "sift") { featureType = FeatureType::SIFT; featureName = "SIFT"; }
        }

        if (argc >= 3)
        {
            std::string arg2 = argv[2];
            std::transform(arg2.begin(), arg2.end(), arg2.begin(), ::tolower);
            if (arg2 == "all")
            {
                maxImages = std::numeric_limits<size_t>::max();
            }
            else
            {
                try { maxImages = static_cast<size_t>(std::stoul(arg2)); }
                catch (...) { maxImages = 1000; }
            }
        }

        if (argc >= 4)
        {
            std::string arg3 = argv[3];
            std::transform(arg3.begin(), arg3.end(), arg3.begin(), ::tolower);
            if (arg3 == "superpixel")
            {
                mode = DescriptorMode::SUPERPIXEL;
                modeName = "SUPERPIXEL";
            }
            else if (arg3 == "global")
            {
                mode = DescriptorMode::GLOBAL;
                modeName = "GLOBAL";
            }
        }

        std::cout << "Program started.\n";
        std::cout << "Feature type: " << featureName << "\n";
        std::cout << "Descriptor mode: " << modeName << "\n";
        if (maxImages == std::numeric_limits<size_t>::max())
            std::cout << "Max images: ALL\n";
        else
            std::cout << "Max images: " << maxImages << "\n";
        std::cout << "Index dir: " << INDEX_DIR << "\n";
        std::cout << "Query img: " << QUERY_IMG << "\n";

        // ---- Load COCO annotations (train + val) ----
        COCOLabelIndex cocoIndex;
        loadCOCOAnnotations(TRAIN_ANN, cocoIndex);
        loadCOCOAnnotations(VAL_ANN, cocoIndex);

        // ---- Collect image paths ----
        std::vector<std::string> imagePaths;
        for (const auto& entry : fs::directory_iterator(INDEX_DIR))
        {
            if (!entry.is_regular_file()) continue;
            if (!isImageFile(entry.path())) continue;
            imagePaths.push_back(entry.path().string());
            if (imagePaths.size() >= maxImages) break;
        }

        std::cout << "Found " << imagePaths.size() << " images to index.\n";
        if (imagePaths.empty())
        {
            std::cerr << "No images found in INDEX_DIR.\n";
            return 1;
        }

        // ---- Multi-threaded descriptor computation ----
        std::vector<ImageJobResult> results(imagePaths.size());
        std::atomic<size_t> nextIndex{0};
        unsigned int numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 4;

        std::cout << "Using " << numThreads << " threads.\n";

        auto worker = [&]()
        {
            while (true)
            {
                size_t idx = nextIndex.fetch_add(1);
                if (idx >= imagePaths.size()) break;
                results[idx] = processImageJob(imagePaths[idx], featureType, mode);
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(numThreads);
        for (unsigned int i = 0; i < numThreads; ++i)
            threads.emplace_back(worker);
        for (auto& t : threads)
            t.join();

        // ---- Build index ----
        ImageIndex index;
        size_t count = 0;
        for (const auto& r : results)
        {
            if (!r.ok) continue;
            index.add(r.path, r.desc);
            count++;
            if (count % 50 == 0)
                std::cout << "Indexed " << count << " images...\n";
        }

        std::cout << "Total indexed images: " << index.filenames.size() << std::endl;
        if (index.filenames.empty())
        {
            std::cerr << "No images successfully indexed.\n";
            return 1;
        }

        // ---- Query descriptor ----
        std::cout << "Loading query image: " << QUERY_IMG << std::endl;
        cv::Mat queryImg = cv::imread(QUERY_IMG, cv::IMREAD_COLOR);
        if (queryImg.empty())
        {
            std::cerr << "Could not read query image.\n";
            return 1;
        }

        cv::Mat queryDesc = buildDescriptor(queryImg, featureType, mode);

        // ---- COCO labels for query ----
        auto queryCats = getCategoriesForImage(cocoIndex, QUERY_IMG);
        std::string queryCatStr = catIdsToString(queryCats, cocoIndex);
        std::unordered_set<int> queryCatSet(queryCats.begin(), queryCats.end());

        if (queryCats.empty())
        {
            std::cerr << "Warning: query image has no COCO categories in annotations.\n";
        }
        else
        {
            std::cout << "Query COCO categories: " << queryCatStr << "\n";
        }

        // ---- Search ----
        const int TOP_K = 5;
        auto matches = index.search(queryDesc, TOP_K);

        std::cout << "\nTop " << TOP_K << " matches:\n";
        for (const auto& [idx, dist] : matches)
        {
            std::cout << "  " << index.filenames[idx] << "  (dist=" << dist << ")\n";
        }

        // ---- Save top-K images ----
        try
        {
            std::string maxStr;
            if (maxImages == std::numeric_limits<size_t>::max())
                maxStr = "all";
            else
                maxStr = std::to_string(maxImages);

            std::string outDir = "../output/" + featureName + "_" + modeName + "_" + maxStr;
            fs::create_directories(outDir);

            std::cout << "\nSaving top " << TOP_K << " matches to: " << outDir << "\n";

            int rank = 1;
            for (const auto& [idx, dist] : matches)
            {
                cv::Mat img = cv::imread(index.filenames[idx], cv::IMREAD_COLOR);
                if (img.empty())
                {
                    std::cerr << "Could not reload " << index.filenames[idx] << " for saving.\n";
                    continue;
                }

                std::string outPath = outDir + "/match_" + std::to_string(rank) + ".jpg";
                if (!cv::imwrite(outPath, img))
                {
                    std::cerr << "Failed to write " << outPath << "\n";
                }
                else
                {
                    std::cout << "  Saved: " << outPath << " (dist=" << dist << ")\n";
                }

                rank++;
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error while saving results: " << e.what() << "\n";
        }

        // ---- CSV with COCO label matching ----
        try
        {
            std::string maxStr;
            if (maxImages == std::numeric_limits<size_t>::max())
                maxStr = "all";
            else
                maxStr = std::to_string(maxImages);

            std::string csvDir  = "../output/csv/";
            fs::create_directories(csvDir);

            std::string csvFile = csvDir + featureName + "_" + modeName + "_" + maxStr + ".csv";
            std::ofstream fout(csvFile);
            if (!fout.is_open())
            {
                std::cerr << "Failed to write CSV: " << csvFile << "\n";
            }
            else
            {
                fout << "method,feature,descriptor_mode,max_images,"
                     << "query_filename,query_categories,"
                     << "match_rank,match_filename,match_categories,shares_label,distance\n";

                std::string methodName = featureName + "_" + modeName + "_" + maxStr;
                std::string queryFname = fs::path(QUERY_IMG).filename().string();

                int rank = 1;
                int correct = 0;

                for (const auto& [idx, dist] : matches)
                {
                    const std::string& fullMatchPath = index.filenames[idx];
                    std::string matchFname = fs::path(fullMatchPath).filename().string();

                    auto matchCats = getCategoriesForImage(cocoIndex, fullMatchPath);
                    std::string matchCatStr = catIdsToString(matchCats, cocoIndex);

                    bool shareLabel = false;
                    for (int c : matchCats)
                    {
                        if (queryCatSet.find(c) != queryCatSet.end())
                        {
                            shareLabel = true;
                            break;
                        }
                    }
                    if (shareLabel) correct++;

                    fout << methodName << ","
                         << featureName << ","
                         << modeName << ","
                         << maxStr << ","
                         << queryFname << ","
                         << "\"" << queryCatStr << "\","
                         << rank << ","
                         << matchFname << ","
                         << "\"" << matchCatStr << "\","
                         << (shareLabel ? 1 : 0) << ","
                         << dist << "\n";

                    rank++;
                }

                fout.close();
                double precisionAtK = (double)correct / (double)TOP_K;
                std::cout << "CSV saved to: " << csvFile << "\n";
                std::cout << "Precision@" << TOP_K << " (COCO category match) = "
                          << precisionAtK << "\n";
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error writing CSV: " << e.what() << "\n";
        }

        std::cout << "Done.\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
