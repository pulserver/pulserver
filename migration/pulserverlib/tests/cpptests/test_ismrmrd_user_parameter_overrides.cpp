#include <cassert>
#include <string>
#include <vector>

#include "ismrmrd_user_parameter_overrides.h"

using GEToIsmrmrd::UserParameterOverride;
using GEToIsmrmrd::apply_ismrmrd_user_parameter_overrides;

int main()
{
    {
        std::string xml = "<ismrmrdHeader><userParameters></userParameters></ismrmrdHeader>";

        UserParameterOverride s;
        s.type = UserParameterOverride::TYPE_STRING;
        s.name = "note";
        s.string_value = "a&b<c>\"'";
        s.long_value = 0;
        s.double_value = 0.0;

        UserParameterOverride l;
        l.type = UserParameterOverride::TYPE_LONG;
        l.name = "counter";
        l.string_value = "";
        l.long_value = 42;
        l.double_value = 0.0;

        UserParameterOverride d;
        d.type = UserParameterOverride::TYPE_DOUBLE;
        d.name = "gain";
        d.string_value = "";
        d.long_value = 0;
        d.double_value = 1.5;

        std::vector<UserParameterOverride> overrides{s, l, d};
        apply_ismrmrd_user_parameter_overrides(xml, overrides);

        assert(xml.find("<userParameterString><name>note</name><value>a&amp;b&lt;c&gt;&quot;&apos;</value></userParameterString>") != std::string::npos);
        assert(xml.find("<userParameterLong><name>counter</name><value>42</value></userParameterLong>") != std::string::npos);
        assert(xml.find("<userParameterDouble><name>gain</name><value>1.5</value></userParameterDouble>") != std::string::npos);
    }

    {
        std::string xml = "<ismrmrdHeader><encoding/></ismrmrdHeader>";

        UserParameterOverride s;
        s.type = UserParameterOverride::TYPE_STRING;
        s.name = "plugin";
        s.string_value = "simplefft";
        s.long_value = 0;
        s.double_value = 0.0;

        std::vector<UserParameterOverride> overrides{s};
        apply_ismrmrd_user_parameter_overrides(xml, overrides);

        assert(xml.find("<userParameters><userParameterString><name>plugin</name><value>simplefft</value></userParameterString></userParameters>") != std::string::npos);
    }

    {
        std::string xml = "<ismrmrdHeader>";

        UserParameterOverride l;
        l.type = UserParameterOverride::TYPE_LONG;
        l.name = "fallback";
        l.string_value = "";
        l.long_value = 7;
        l.double_value = 0.0;

        std::vector<UserParameterOverride> overrides{l};
        apply_ismrmrd_user_parameter_overrides(xml, overrides);

        assert(xml.find("<userParameters><userParameterLong><name>fallback</name><value>7</value></userParameterLong></userParameters>") != std::string::npos);
    }

    return 0;
}
