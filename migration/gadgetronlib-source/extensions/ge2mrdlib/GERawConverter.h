/** @file GERawConverter.h */
#ifndef GE_RAW_CONVERTER_H
#define GE_RAW_CONVERTER_H

#include <fstream>
#include <vector>

// ISMRMRD
#include "ismrmrd/ismrmrd.h"

// Local
#include "SequenceConverter.h"
#include "GenericConverter.h"
#include "ismrmrd_user_parameter_overrides.h"

// Libxml2 forward declarations
struct _xmlDoc;
struct _xmlNode;

namespace GEToIsmrmrd {

struct logstream {
    logstream(bool enable) : enabled(enable) {}
    bool enabled;
};

template <typename T>
logstream& operator<<(logstream& s, T const& v)
{
    if (s.enabled) { std::clog << v; }
    return s;
}

logstream& operator<<(logstream& s, std::ostream& (*f)(std::ostream&));

enum GE_RAW_TYPES
{
   SCAN_ARCHIVE_RAW_TYPE = 0,
   PFILE_RAW_TYPE = 1,
   MISC_RAW_TYPE = 99
};


class GERawConverter
{
public:

    // Existing constructor
    GERawConverter(const std::string& pfilepath, const std::string& classname, bool logging=false);

    // New constructor: accepts lxData and processingControl directly, assumes GenericConverter and loads default.xsl
    GERawConverter(bool logging=false);

    std::shared_ptr<SequenceConverter> getConverter();

    void setHeader(const boost::shared_ptr<const GERecon::Legacy::LxDownloadData>& lxData);
    
    void useStylesheetFilename(const std::string& filename);
    void useStylesheetStream(std::ifstream& stream);
    void useStylesheetString(const std::string& sheet);

    // Optional overrides injected into final ISMRMRD XML output.
    void clearIsmrmrdUserParameterOverrides();
    void setIsmrmrdUserParameterString(const std::string& name, const std::string& value);
    void setIsmrmrdUserParameterLong(const std::string& name, long long value);
    void setIsmrmrdUserParameterDouble(const std::string& name, double value);

    std::string getIsmrmrdXMLHeader();

    ISMRMRD::Acquisition getAcquisition(int dataIndex,
                                        const GERecon::Acquisition::ProgrammableControlPacket& programmableControlPacket, 
                                        const GERecon::Acquisition::FramePointer& frame,
                                        const GERecon::Control::ProcessingControl& control);

    std::vector<ISMRMRD::Acquisition> getAcquisitions(unsigned int view_num);

    std::string getReconConfigName(void);

    std::string ge_header_to_xml(GERecon::Legacy::LxDownloadDataPointer lxData,
                                 GERecon::Control::ProcessingControlPointer processingControl);
private:
    // Non-copyable
    GERawConverter(const GERawConverter& other);
    GERawConverter& operator=(const GERawConverter& other);

    bool validateConfig(std::shared_ptr<struct _xmlDoc> config_doc);
    bool trySequenceMapping(std::shared_ptr<struct _xmlDoc> doc, struct _xmlNode* mapping);

    void applyIsmrmrdUserParameterOverrides(std::string& ismrmrd_header) const;

    std::string psdname_;
    std::string recon_config_;
    std::string stylesheet_;
    std::vector<UserParameterOverride> user_parameter_overrides_;

    GERecon::Legacy::PfilePointer pfile_;

    GERecon::ScanArchivePointer scanArchive_;
    GERecon::Legacy::LxDownloadDataPointer lxData_;
    GERecon::Control::ProcessingControlPointer processingControl_;
    int rawObjectType_; // to allow reference to a P-File or ScanArchive object
    std::shared_ptr<GEToIsmrmrd::SequenceConverter> converter_;

    logstream log_;
};

} // namespace GEToIsmrmrd

#endif  // GE_RAW_CONVERTER_H
