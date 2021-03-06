// This is the std_common.i from swig 2.0.11.
// Modifications:
// - use SvPVutf8 instead of SvPV in SwigSvToString
// - use SvUTF8_on in SwigSvFromString

/* -----------------------------------------------------------------------------
 * std_common.i
 *
 * SWIG typemaps for STL - common utilities
 * ----------------------------------------------------------------------------- */

%include <std/std_except.i>

%apply size_t { std::size_t };

%{
#include <string>

double SwigSvToNumber(SV* sv) {
    return SvIOK(sv) ? double(SvIVX(sv)) : SvNVX(sv);
}
std::string SwigSvToString(SV* sv) {
    STRLEN len;
    char *ptr = SvPVutf8(sv, len);
    return std::string(ptr, len);
}
void SwigSvFromString(SV* sv, const std::string& s) {
    sv_setpvn(sv,s.data(),s.size());
    SvUTF8_on(sv);
}
%}
