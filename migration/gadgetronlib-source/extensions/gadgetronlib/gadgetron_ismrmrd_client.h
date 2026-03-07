#pragma once

#include <boost/asio.hpp>

#include <ismrmrd/ismrmrd.h>
#include <ismrmrd/dataset.h>
#include <ismrmrd/meta.h>
#include <ismrmrd/xml.h>
#include <ismrmrd/waveform.h>

#include <map>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>

#include "NHLBICompression.h"
#include "reader_wait_state.h"

using namespace NHLBI;
using boost::asio::ip::tcp;

enum GadgetronMessageID {
    GADGET_MESSAGE_INT_ID_MIN                             =   0,
    GADGET_MESSAGE_CONFIG_FILE                            =   1,
    GADGET_MESSAGE_CONFIG_SCRIPT                          =   2,
    GADGET_MESSAGE_PARAMETER_SCRIPT                       =   3,
    GADGET_MESSAGE_CLOSE                                  =   4,
    GADGET_MESSAGE_TEXT                                   =   5,
    GADGET_MESSAGE_QUERY                                  =   6,
    GADGET_MESSAGE_RESPONSE                               =   7,
    GADGET_MESSAGE_INT_ID_MAX                             = 999,
    GADGET_MESSAGE_EXT_ID_MIN                             = 1000,
    GADGET_MESSAGE_ACQUISITION                            = 1001, /**< DEPRECATED */
    GADGET_MESSAGE_NEW_MEASUREMENT                        = 1002, /**< DEPRECATED */
    GADGET_MESSAGE_END_OF_SCAN                            = 1003, /**< DEPRECATED */
    GADGET_MESSAGE_IMAGE_CPLX_FLOAT                       = 1004, /**< DEPRECATED */
    GADGET_MESSAGE_IMAGE_REAL_FLOAT                       = 1005, /**< DEPRECATED */
    GADGET_MESSAGE_IMAGE_REAL_USHORT                      = 1006, /**< DEPRECATED */
    GADGET_MESSAGE_EMPTY                                  = 1007, /**< DEPRECATED */
    GADGET_MESSAGE_ISMRMRD_ACQUISITION                    = 1008,
    GADGET_MESSAGE_ISMRMRD_IMAGE_CPLX_FLOAT               = 1009, /**< DEPRECATED */
    GADGET_MESSAGE_ISMRMRD_IMAGE_REAL_FLOAT               = 1010, /**< DEPRECATED */
    GADGET_MESSAGE_ISMRMRD_IMAGE_REAL_USHORT              = 1011, /**< DEPRECATED */
    GADGET_MESSAGE_DICOM                                  = 1012, /**< DEPRECATED */
    GADGET_MESSAGE_CLOUD_JOB                              = 1013,
    GADGET_MESSAGE_GADGETCLOUD_JOB                        = 1014,
    GADGET_MESSAGE_ISMRMRD_IMAGEWITHATTRIB_CPLX_FLOAT     = 1015, /**< DEPRECATED */
    GADGET_MESSAGE_ISMRMRD_IMAGEWITHATTRIB_REAL_FLOAT     = 1016, /**< DEPRECATED */
    GADGET_MESSAGE_ISMRMRD_IMAGEWITHATTRIB_REAL_USHORT    = 1017, /**< DEPRECATED */
    GADGET_MESSAGE_DICOM_WITHNAME                         = 1018,
    GADGET_MESSAGE_DEPENDENCY_QUERY                       = 1019,
    GADGET_MESSAGE_ISMRMRD_IMAGE_REAL_SHORT               = 1020, /**< DEPRECATED */
    GADGET_MESSAGE_ISMRMRD_IMAGEWITHATTRIB_REAL_SHORT     = 1021, /**< DEPRECATED */
    GADGET_MESSAGE_ISMRMRD_IMAGE                          = 1022,
    GADGET_MESSAGE_RECONDATA                              = 1023,
    GADGET_MESSAGE_ISMRMRD_WAVEFORM                       = 1026,
    GADGET_MESSAGE_EXT_ID_MAX                             = 4096
};

struct NoiseStatistics
{
    bool status;
    uint16_t channels;
    float sigma_min;
    float sigma_max;
    float sigma_mean;
    float noise_dwell_time_us;
};

class GadgetronClientMessageReader
{
public:
    virtual ~GadgetronClientMessageReader() {}

    /**
    Function must be implemented to read a specific message.
    */
    virtual void read(tcp::socket* s) = 0;

};

class GadgetronClientConnector
{
public:
    GadgetronClientConnector();
    virtual ~GadgetronClientConnector();

    double compression_ratio();
    double get_bytes_transmitted();
    void set_timeout(unsigned int t);

    void read_task();
    void wait();
    bool wait_for(unsigned int timeout_ms);
    void connect(std::string hostname, std::string port);

    void send_gadgetron_close();
    void send_gadgetron_info_query(const std::string &query, uint64_t correlation_id = 0);
    void send_gadgetron_configuration_file(std::string config_xml_name);
    void send_gadgetron_configuration_script(std::string xml_string);
    void send_gadgetron_parameters(std::string xml_string);

    void send_ismrmrd_acquisition(ISMRMRD::Acquisition& acq);
    void send_ismrmrd_compressed_acquisition_precision(ISMRMRD::Acquisition& acq, unsigned int compression_precision);
    void send_ismrmrd_compressed_acquisition_tolerance(ISMRMRD::Acquisition& acq, float compression_tolerance, NoiseStatistics& stat);
    void send_ismrmrd_zfp_compressed_acquisition_precision(ISMRMRD::Acquisition& acq, unsigned int compression_precision);
    void send_ismrmrd_zfp_compressed_acquisition_tolerance(ISMRMRD::Acquisition& acq, float compression_tolerance, NoiseStatistics& stat);
    void send_ismrmrd_waveform(ISMRMRD::Waveform& wav);

    void register_reader(unsigned short slot, std::shared_ptr<GadgetronClientMessageReader> r);

protected:
    typedef std::map<unsigned short, std::shared_ptr<GadgetronClientMessageReader> > maptype;
    GadgetronClientMessageReader* find_reader(unsigned short r);

    boost::asio::io_service io_service;
    tcp::socket* socket_;
    std::thread reader_thread_;
    maptype readers_;
    unsigned int timeout_ms_;
    ReaderWaitState reader_state_;
    double header_bytes_sent_;
    double uncompressed_bytes_sent_;
    double compressed_bytes_sent_;
};

void send_ismrmrd_acq(GadgetronClientConnector& con, ISMRMRD::Acquisition& acq_tmp, 
    unsigned int compression_precision, bool use_zfp_compression, float compression_tolerance, NoiseStatistics& noise_stats);

/**
 * Stand-alone callable gadgetron ISMRM client function
 * Adapted from: https://github.com/gadgetron/gadgetron/blob/master/apps/clients/gadgetron_ismrmrd_client/gadgetron_ismrmrd_client.cpp
 */
namespace Gadgetron
{
    int gadgetron_ismrmrd_client(const std::string& out_filename,
                                 const std::string& hdf5_out_group,
                                 const std::string& in_filename,
                                 const std::string& hdf5_in_group,
                                 const std::string& host_name = "localhost",
                                 const std::string& port = "9002",
                                 const std::string& config_file = "default.xml",
                                 const std::string& config_file_local = "",
                                 const std::string& config_xml_local = "",
                                 const unsigned int loops = 1,
                                 const unsigned int timeout_ms = 10000,
                                 const std::string& out_fileformat = "h5",
                                 const unsigned int compression_precision = 0,
                                 const bool query = false,
                                 const std::string& info = "",
                                 const float compression_tolerance = 0.0,
                                 const bool use_zfp_compression = false,
                                 const bool verbose = false);
}
