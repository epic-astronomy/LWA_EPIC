# [0.9.0](https://github.com/epic-astronomy/LWA_EPIC/compare/v0.8.0...v0.9.0) (2024-04-07)


### Bug Fixes

* corrected the source extraction index ([5f51879](https://github.com/epic-astronomy/LWA_EPIC/commit/5f51879ff257b4a00db74f0ecae4ba145f97d034))


### Features

* added command to save a quick look png image for each fits file ([17ceff9](https://github.com/epic-astronomy/LWA_EPIC/commit/17ceff96588515137d2f6cbc2c6c08ba403e6479))



# [0.8.0](https://github.com/epic-astronomy/LWA_EPIC/compare/v0.7.2...v0.8.0) (2024-04-03)


### Bug Fixes

* changed the disk saver to write same filenames to db and disk ([95a8fcf](https://github.com/epic-astronomy/LWA_EPIC/commit/95a8fcf47de2cfc3437a0f8eead7bb0cd1dc6c81))
* fixed the fits header to reflect the correct image duration ([424b134](https://github.com/epic-astronomy/LWA_EPIC/commit/424b1343194051ad88ec4ed932fe264bd1e0f1e0))


### Features

* remove autocorrelations ([23e5377](https://github.com/epic-astronomy/LWA_EPIC/commit/23e537747d34d735db65630e753f100dd8b85cd1))



## [0.7.2](https://github.com/epic-astronomy/LWA_EPIC/compare/v0.7.1...v0.7.2) (2024-03-30)


### Bug Fixes

* changed the obs time to center of the image duration instead of the end ([8999396](https://github.com/epic-astronomy/LWA_EPIC/commit/89993967776e18030cb3bd268fe3c9d0fe2e03c4))
* corrected the pixel index calculation while extracting pixels ([fb094ad](https://github.com/epic-astronomy/LWA_EPIC/commit/fb094ad4e8897aa2ac4155b83e8648fc886ecdb1))


### Performance Improvements

* Increased the buffer size to accomodate db ingestion delays ([2ef38a2](https://github.com/epic-astronomy/LWA_EPIC/commit/2ef38a23a5770b60645d2c7cfcd478cc29f46875))
* increased the buufer count to accomodate db ingestion delays ([8ba8140](https://github.com/epic-astronomy/LWA_EPIC/commit/8ba8140fd56d6e224ef1a58ae4f936b2f06ab279))



## [0.7.1](https://github.com/epic-astronomy/LWA_EPIC/compare/v0.7.0...v0.7.1) (2024-03-27)


### Bug Fixes

* Removed system libraries from the linking list ([0711403](https://github.com/epic-astronomy/LWA_EPIC/commit/07114036480a36e1e0765b959c9b8c902d5b757d))



# [0.7.0](https://github.com/epic-astronomy/LWA_EPIC/compare/v0.6.0...v0.7.0) (2024-03-27)


### Bug Fixes

* changed the precision for time from unix epoch from float to int ([0739e43](https://github.com/epic-astronomy/LWA_EPIC/commit/0739e4337106bf0075f82f284056586b1457c5b2))


### Features

* Added ability to specify output directory to store accumulated files ([99045b6](https://github.com/epic-astronomy/LWA_EPIC/commit/99045b6ba4a4f185c3193732a5c9d7f2ee2efa0b))



