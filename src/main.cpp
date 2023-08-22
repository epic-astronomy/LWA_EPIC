/*
 Copyright (c) 2023 The EPIC++ authors

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

// clang-format off
// IMPORTANT: pqxx MUST be the first header.
// A weird interaction with other libraries
// makes the compiler yell at us!
#include <pqxx/pqxx>
// clang-format on

#include "raft_kernels/epic_executor.hpp"

namespace py = pybind11;
// #define _VMA_ 1
int main(int argc, char** argv) {
  py::scoped_interpreter guard{};
  py::gil_scoped_release release;

  FLAGS_logtostderr = "1";
  google::InitGoogleLogging(argv[0]);
  google::EnableLogCleaner(3);

  RunEpic(argc, argv);

  return EXIT_SUCCESS;
}
