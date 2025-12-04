#include <vector>

class SuperDuperPixel
{
public:
	SuperDuperPixel(int superpixel, std::vector<float> average_or_histogram, int pixel_count);
	float distance_from(const std::vector<float>& average_or_histogram);
	void add_superpixel(int superpixel, const std::vector<float>& average_or_histogram, int pixel_count);
	std::vector<int> get_superpixels();
	void operator+=(const SuperDuperPixel* other);
private:
	std::vector<int> superpixels;
	std::vector<float> average_or_histogram;
	int pixel_count;
};
