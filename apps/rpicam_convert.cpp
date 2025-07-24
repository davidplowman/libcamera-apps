#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

#include <sys/mman.h>

#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/formats.h>
#include <libcamera/stream.h>

#include <libraw/libraw.h>

#include "core/dma_heaps.hpp"
#include "core/still_options.hpp"
#include "core/stream_info.hpp"
#include "image/image.hpp"

using namespace libcamera;

struct Buffer
{
	std::unique_ptr<FrameBuffer> fb;
	Span<uint8_t> span;
};

Buffer allocate_frame_buffer(StreamConfiguration &config, DmaHeap &heap)
{
	Buffer buffer;
	UniqueFD fd = heap.alloc("rpicam-convert", config.frameSize);

	if (!fd.isValid())
		throw std::runtime_error("failed to allocate capture buffers for stream");

	std::vector<FrameBuffer::Plane> plane(1);
	plane[0].fd = libcamera::SharedFD(std::move(fd));
	plane[0].offset = 0;
	plane[0].length = config.frameSize;

	buffer.fb =std::make_unique<FrameBuffer>(plane);
	void *memory = mmap(NULL, config.frameSize, PROT_READ | PROT_WRITE, MAP_SHARED, plane[0].fd.get(), 0);
	buffer.span = Span<uint8_t>(static_cast<uint8_t *>(memory), config.frameSize);

	return buffer;
}

static std::mutex mtx;
static std::condition_variable cv;
static bool request_completed = false;

void request_complete(Request *request)
{
    {
        std::lock_guard<std::mutex> lock(mtx);
        request_completed = true;
    }
    cv.notify_one();
}

StreamInfo get_stream_info(StreamConfiguration const &cfg)
{
	StreamInfo info;
	info.width = cfg.size.width;
	info.height = cfg.size.height;
	info.stride = cfg.stride;
	info.pixel_format = cfg.pixelFormat;
	info.colour_space = cfg.colorSpace;
	return info;
}

std::string get_bayer_order(LibRaw *raw)
{
    const libraw_iparams_t& idata = raw->imgdata.idata;
	char bayer_order[5] = { 0, 0, 0, 0, 0 };

	// Yes, this really does make a string like "BGGR" etc.
	for (int i = 0, filters = idata.filters; i < 4; i++, filters >>= 2)
		bayer_order[i] = idata.cdesc[filters & 3];

	return std::string(bayer_order);
}

void copy_raw_to_input(LibRaw *raw, Buffer &input_buffer, StreamConfiguration const &input_config)
{
	uint8_t *iptr = reinterpret_cast<uint8_t *>(raw->imgdata.rawdata.raw_image);
	uint8_t *optr = &input_buffer.span[0];
	int istride = raw->imgdata.sizes.raw_pitch, ostride = input_config.stride;
	int nbytes = std::min(istride, ostride);

	for (unsigned int h = 0; h < input_config.size.height; h++, iptr += istride, optr += ostride)
		memcpy(optr, iptr, nbytes);
}

PixelFormat get_raw_format(std::string const &bayer_order, int bps)
{
	static std::map<std::string, std::vector<PixelFormat>> table = {
		{ "BGGR", { formats::SBGGR16, formats::SBGGR14, formats::SBGGR12, formats::SBGGR10 } },
		{ "RGGB", { formats::SRGGB16, formats::SRGGB14, formats::SRGGB12, formats::SRGGB10 } },
		{ "GRBG", { formats::SGRBG16, formats::SGRBG14, formats::SGRBG12, formats::SGRBG10 } },
		{ "GBRG", { formats::SGBRG16, formats::SGBRG14, formats::SGBRG12, formats::SGBRG10 } }
	};
	static std::map<int, int> bpsToIndex = { { 16, 0 }, { 14, 1 }, { 12, 2 }, { 10, 3 } };
	auto it = bpsToIndex.find(bps);
	if (it == bpsToIndex.end())
		throw std::runtime_error("Bit depth " + std::to_string(bps) + " not supported");

	return table[bayer_order][it->second];
}

LibRaw *load_raw_file(char const *filename)
{
	LibRaw *raw = new LibRaw();
	if (raw->open_file(filename) < 0)
		throw std::runtime_error("Failed to load file " + std::string(filename));
	raw->unpack();

	// Let's print out some handy info about the raw file!
    const libraw_image_sizes_t& sizes = raw->imgdata.sizes;
    const libraw_colordata_t& color = raw->imgdata.color;

	std::cout << "DNG file info for " << filename << ":" << std::endl;
    std::cout << "    Width:  " << sizes.width << std::endl;
    std::cout << "    Height: " << sizes.height << std::endl;

	std::cout << "    Shutter: " << raw->imgdata.other.shutter << std::endl;
	std::cout << "    Analogue gain: " << raw->imgdata.other.iso_speed / 100.0 << std::endl;

	// The actual black level is here, not in color.black.
    std::cout << "    Black level: " << color.cblack[6] << std::endl;
	std::cout << "    Raw bps: " << color.raw_bps << std::endl;
	std::cout << "    Bayer order: " << get_bayer_order(raw) << std::endl;

    // As Shot Neutral has been converted to colour gains
    std::cout << "    Colour gains: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << color.cam_mul[i] << " ";
    }
    std::cout << std::endl;

    // Colour matrix (cam to sRGB)
    std::cout << "    Cam to sRGB matrix:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "        ";
        for (int j = 0; j < 3; j++)
            std::cout << color.rgb_cam[i][j] << " ";
        std::cout << std::endl;
    }

	return raw;
}

int main(int argc, char *argv[])
{
	if (argc != 4) {
		fprintf(stderr, "Expected 3 arguments: %s <input.dng> <tuning.json> <output.jpg>\n", argv[0]);
		return -1;
	}
	char const *input_filename = argv[1];
	char const *tuning_filename = argv[2];
	char const *output_filename = argv[3];

	std::unique_ptr<CameraManager> camera_manager = std::make_unique<CameraManager>();
	int ret = camera_manager->start();
	if (ret)
		throw std::runtime_error("camera manager failed to start, code " + std::to_string(-ret));

	auto memory_cameras = camera_manager->memoryCameras();

	// Get a memory camera that will use the PiSP for processing, and the tuning file
	// will define what algorithms will run.
	std::shared_ptr<Camera> memory_camera = camera_manager->getMemoryCamera("rpi/pisp", std::string(tuning_filename));

	LibRaw *raw = load_raw_file(input_filename);

	std::vector<StreamRole> stream_roles = { StreamRole::RawInput, StreamRole::StillCapture };
	std::unique_ptr<CameraConfiguration> camera_configuration = memory_camera->generateConfiguration(stream_roles);

	StreamConfiguration &input_config = camera_configuration->at(0);
	StreamConfiguration &output_config = camera_configuration->at(1);

	// Set up the sensor config to look like our raw file.
	auto sizes = raw->imgdata.sizes;
	int bps = raw->imgdata.color.raw_bps;
	camera_configuration->sensorConfig.emplace();
	camera_configuration->sensorConfig->bitDepth = bps;
	camera_configuration->sensorConfig->outputSize.width = sizes.width;
	camera_configuration->sensorConfig->outputSize.height = sizes.height;

	input_config.size.width = sizes.width;
	input_config.size.height = sizes.height;
	input_config.pixelFormat = get_raw_format(get_bayer_order(raw), bps);

	output_config.size.width = sizes.width;
	output_config.size.height = sizes.height;
	output_config.pixelFormat = formats::YUV420;
	output_config.colorSpace = ColorSpace::Sycc;

	memory_camera->acquire();
	memory_camera->requestCompleted.connect(request_complete);
	memory_camera->configure(camera_configuration.get());

	// Let's allocate some buffers.
	DmaHeap dma_heap;
	Buffer input_buffer = allocate_frame_buffer(input_config, dma_heap);
	Buffer output_buffer = allocate_frame_buffer(output_config, dma_heap);

	// Copy raw data to input buffer.
	copy_raw_to_input(raw, input_buffer, input_config);

	// And put them in a request. Also add the colour gains, which AWB should pick up,
	// the CCM and exposure info which algorithms will use to pick appropriate parameters.
	std::unique_ptr<Request> request = memory_camera->createRequest();
	request->addBuffer(input_config.stream(), input_buffer.fb.get());
	request->addBuffer(output_config.stream(), output_buffer.fb.get());
	float red_gain = raw->imgdata.color.cam_mul[0];
	float blue_gain = raw->imgdata.color.cam_mul[2];
	request->controls().set(controls::ColourGains,
							Span<const float, 2>({ red_gain, blue_gain }));
	std::array<float, 9> ccm;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            ccm[3 * i + j] = std::min(raw->imgdata.color.rgb_cam[i][j] * 1.0, 65535.0);
	}
	request->controls().set(controls::ColourCorrectionMatrix, ccm);
	unsigned int shutter_us = raw->imgdata.other.shutter * 1000000;
	request->controls().set(controls::ExposureTime, shutter_us);
	request->controls().set(controls::AnalogueGain, raw->imgdata.other.iso_speed / 100.0);

	memory_camera->start();

	if (memory_camera->queueRequest(request.get()) < 0)
		throw std::runtime_error("Failed to queue request");

	{
		std::unique_lock<std::mutex> lock(mtx);
		if (cv.wait_for(lock, std::chrono::seconds(1), [] { return request_completed; }))
			/* Request completed successfully. */;
		else
			throw std::runtime_error("Request timed out");
	}

	// Save a JPEG.
	StreamInfo info = get_stream_info(output_config);
	const std::vector<Span<uint8_t>> mem = { output_buffer.span };
	StillOptions options;
	options.Set().thumb_quality = 0;
	options.Set().quality = 100;
	jpeg_save(mem, info, request->metadata(), output_filename, "DNG", &options);
	std::cout << "Wrote converted raw image to " << output_filename << std::endl;

	memory_camera->stop();
	memory_camera->release();
	memory_camera.reset();

	camera_manager->stop();
	camera_manager.reset();

	std::cout << argv[0] << " finished!" << std::endl;
	return 0;
}
