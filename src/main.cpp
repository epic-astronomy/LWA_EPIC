// IMPORTANT: pqxx MUST be the first header. A weird interaction with other libraries
// makes the compiler yell at us!
// clang-format off
#include <pqxx/pqxx>
// clang-format on

#include "raft_kernels/epic_executor.hpp"

namespace py = pybind11;
// #define _VMA_ 1
int
main(int argc, char** argv)
{
    py::scoped_interpreter guard{};
    py::gil_scoped_release release;

    FLAGS_logtostderr = 1;
    google::InitGoogleLogging(argv[0]);
    google::EnableLogCleaner(3);

    run_epic(argc, argv);

    return EXIT_SUCCESS;
}
