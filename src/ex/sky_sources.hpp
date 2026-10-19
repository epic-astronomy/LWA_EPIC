// Directions of the bright compact sources (A-team, Sun) at LWA-SV, for their
// subtraction from the visibilities (FxImager peeling).
//
// Low-precision ephemeris, good to about 0.01 deg (the synthesized beam is about
// 2 deg): J2000 positions precessed to the date (IAU 1976), Greenwich mean
// sidereal time (IAU 1982), the Sun from the Astronomical Almanac's low-precision
// formulae; no nutation, aberration or refraction.
#ifndef SRC_EX_SKY_SOURCES_HPP_
#define SRC_EX_SKY_SOURCES_HPP_

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace sky {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;
// LWA-SV (as lsl.common.stations.lwasv)
constexpr double kLatDeg = 34.348361;
constexpr double kLonDeg = -106.885778;

struct Source {
  std::string name;
  double ra_deg, dec_deg;  // J2000; ignored for the Sun
  bool sun{false};
};

inline const std::vector<Source>& BrightSources() {
  static const std::vector<Source> s = {
      {"CygA", 299.868167, 40.733917}, {"CasA", 350.850000, 58.815000},
      {"TauA", 83.633083, 22.014500},  {"VirA", 187.705917, 12.391111},
      {"Sun", 0, 0, true}};
  return s;
}

inline double JulianDate(double p_unix) { return p_unix / 86400.0 + 2440587.5; }

/// Greenwich mean sidereal time (rad)
inline double Gmst(double p_unix) {
  const double d = JulianDate(p_unix) - 2451545.0, t = d / 36525.0;
  double g = 280.46061837 + 360.98564736629 * d + 0.000387933 * t * t - t * t * t / 38710000.0;
  g = std::fmod(g, 360.0);
  return (g < 0 ? g + 360.0 : g) * kDeg;
}

/// J2000 (ra, dec) to the mean equator of the date (rad)
inline void Precess(double p_unix, double& p_ra, double& p_dec) {
  const double t = (JulianDate(p_unix) - 2451545.0) / 36525.0;
  const double as = kDeg / 3600.0;
  const double zeta = (2306.2181 * t + 0.30188 * t * t + 0.017998 * t * t * t) * as;
  const double z = (2306.2181 * t + 1.09468 * t * t + 0.018203 * t * t * t) * as;
  const double th = (2004.3109 * t - 0.42665 * t * t - 0.041833 * t * t * t) * as;
  const double a = std::cos(p_dec) * std::sin(p_ra + zeta);
  const double b = std::cos(th) * std::cos(p_dec) * std::cos(p_ra + zeta) - std::sin(th) * std::sin(p_dec);
  const double c = std::sin(th) * std::cos(p_dec) * std::cos(p_ra + zeta) + std::cos(th) * std::sin(p_dec);
  p_ra = std::atan2(a, b) + z;
  p_dec = std::asin(std::fmax(-1.0, std::fmin(1.0, c)));
}

/// The Sun's (ra, dec) of the date (rad)
inline void Sun(double p_unix, double& p_ra, double& p_dec) {
  const double n = JulianDate(p_unix) - 2451545.0;
  const double L = (280.460 + 0.9856474 * n) * kDeg;
  const double g = (357.528 + 0.9856003 * n) * kDeg;
  const double lam = L + (1.915 * std::sin(g) + 0.020 * std::sin(2 * g)) * kDeg;
  const double eps = (23.439 - 0.0000004 * n) * kDeg;
  p_ra = std::atan2(std::cos(eps) * std::sin(lam), std::cos(lam));
  p_dec = std::asin(std::sin(eps) * std::sin(lam));
}

/// Unit vector (east, north, up) at LWA-SV of (ra, dec) of the date at p_unix
inline std::array<double, 3> Enu(double p_unix, double p_ra, double p_dec) {
  const double lat = kLatDeg * kDeg;
  const double h = Gmst(p_unix) + kLonDeg * kDeg - p_ra;  // hour angle
  return {-std::cos(p_dec) * std::sin(h),
          std::sin(p_dec) * std::cos(lat) - std::cos(p_dec) * std::cos(h) * std::sin(lat),
          std::sin(p_dec) * std::sin(lat) + std::cos(p_dec) * std::cos(h) * std::cos(lat)};
}

/// Direction (east, north, up) of source p_s at p_unix
inline std::array<double, 3> Direction(const Source& p_s, double p_unix) {
  double ra, dec;
  if (p_s.sun) {
    Sun(p_unix, ra, dec);
  } else {
    ra = p_s.ra_deg * kDeg;
    dec = p_s.dec_deg * kDeg;
    Precess(p_unix, ra, dec);
  }
  return Enu(p_unix, ra, dec);
}

}  // namespace sky

#endif  // SRC_EX_SKY_SOURCES_HPP_
