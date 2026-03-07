#ifndef ISMRMRD_USER_PARAMETER_OVERRIDES_H
#define ISMRMRD_USER_PARAMETER_OVERRIDES_H

#include <string>
#include <vector>

namespace GEToIsmrmrd {

struct UserParameterOverride {
    enum ValueType {
        TYPE_STRING,
        TYPE_LONG,
        TYPE_DOUBLE,
    };

    ValueType type;
    std::string name;
    std::string string_value;
    long long long_value;
    double double_value;
};

std::string escape_xml(const std::string& input);
void apply_ismrmrd_user_parameter_overrides(
    std::string& ismrmrd_header,
    const std::vector<UserParameterOverride>& overrides);

} // namespace GEToIsmrmrd

#endif
