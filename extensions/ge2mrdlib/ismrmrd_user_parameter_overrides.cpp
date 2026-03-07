#include "ismrmrd_user_parameter_overrides.h"

#include <sstream>

namespace GEToIsmrmrd {

std::string escape_xml(const std::string& input)
{
    std::string out;
    out.reserve(input.size());
    for (std::string::const_iterator it = input.begin(); it != input.end(); ++it) {
        switch (*it) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            case '\'': out += "&apos;"; break;
            default: out += *it; break;
        }
    }
    return out;
}

void apply_ismrmrd_user_parameter_overrides(
    std::string& ismrmrd_header,
    const std::vector<UserParameterOverride>& overrides)
{
    if (overrides.empty()) {
        return;
    }

    std::ostringstream entries;
    for (std::vector<UserParameterOverride>::const_iterator it = overrides.begin();
         it != overrides.end();
         ++it)
    {
        if (it->type == UserParameterOverride::TYPE_STRING) {
            entries << "<userParameterString><name>" << escape_xml(it->name)
                    << "</name><value>" << escape_xml(it->string_value)
                    << "</value></userParameterString>";
        } else if (it->type == UserParameterOverride::TYPE_LONG) {
            entries << "<userParameterLong><name>" << escape_xml(it->name)
                    << "</name><value>" << it->long_value
                    << "</value></userParameterLong>";
        } else {
            entries << "<userParameterDouble><name>" << escape_xml(it->name)
                    << "</name><value>" << it->double_value
                    << "</value></userParameterDouble>";
        }
    }

    const std::string close_user_params = "</userParameters>";
    const std::string close_header = "</ismrmrdHeader>";
    const std::string entry_xml = entries.str();

    std::string::size_type pos = ismrmrd_header.find(close_user_params);
    if (pos != std::string::npos) {
        ismrmrd_header.insert(pos, entry_xml);
        return;
    }

    pos = ismrmrd_header.find(close_header);
    if (pos != std::string::npos) {
        ismrmrd_header.insert(pos, "<userParameters>" + entry_xml + "</userParameters>");
        return;
    }

    ismrmrd_header += "<userParameters>" + entry_xml + "</userParameters>";
}

} // namespace GEToIsmrmrd
