#include "SuperDuperPixel.hpp"
#include <assert.h>

SuperDuperPixel::SuperDuperPixel(int superpixel, std::vector<float> average_or_histogram, int pixel_count)
{
	this->superpixels.push_back(superpixel);
	this->average_or_histogram = average_or_histogram;
	this->pixel_count = pixel_count;
}

float SuperDuperPixel::distance_from(const std::vector<float>& average_or_histogram)
{
	assert(this->average_or_histogram.size() == average_or_histogram->size());
	float dist = 0;
	for (int color_channel = 0; color_channel < this->average_or_histogram.size(); color_channel += 1)
	{
		float diff = this->average_or_histogram[color_channel] - average_or_histogram[color_channel];
		// OpenCV SLIC algorithm square diff before adding it to dist.
		// dist += diff * diff;
		// Just take absolute value to do mahnattan distance instead.
		dist += abs(diff);
	}
	// Just use manhattan distance here.
	// Could do this to be more precise (euclidian distance, would also need to square the diff above), but OpenCV
	// SLIC algorithm doesn't use it either.
	// dist = sqrt(dist);
	return dist;
}

void SuperDuperPixel::add_superpixel(int superpixel, const std::vector<float>& average_or_histogram, int pixel_count)
{
	assert(this->average_or_histogram.size() == average_or_histogram->size());
	this->superpixels.push_back(superpixel);
	int new_pixel_count = this->pixel_count + pixel_count;
	for (int color_channel = 0; color_channel < this->average_or_histogram.size(); color_channel += 1)
	{
		float this_sum = this->average_or_histogram[color_channel] * this->pixel_count;
		float other_sum = average_or_histogram[color_channel] * pixel_count;
		this->average_or_histogram[color_channel] = (this_sum + other_sum) / new_pixel_count;
	}
	this->pixel_count = new_pixel_count;
}

std::vector<int> SuperDuperPixel::get_superpixels() { return this->superpixels; }

void SuperDuperPixel::operator+=(const SuperDuperPixel* other)
{
	assert(this->average_or_histogram.size() == other.average_or_histogram.size());
	this->superpixels.insert(this->superpixels.end(), other->superpixels.begin(), other->superpixels.end());
	int new_pixel_count = this->pixel_count + other->pixel_count;
	for (int color_channel = 0; color_channel < this->average_or_histogram.size(); color_channel += 1)
	{
		float this_sum = this->average_or_histogram[color_channel] * this->pixel_count;
		float other_sum = other->average_or_histogram[color_channel] * other->pixel_count;
		this->average_or_histogram[color_channel] = (this_sum + other_sum) / new_pixel_count;
	}
	this->pixel_count = new_pixel_count;
}
