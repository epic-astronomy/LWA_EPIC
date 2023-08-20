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

#ifndef SRC_EX_EXCEPTIONS_HPP_
#define SRC_EX_EXCEPTIONS_HPP_

#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

// template<typename T = void>
// class ExceptionPrinter
// {
//   protected:
//     std::stringstream m_err_stream;

//   public:
//     virtual void print() const
//     {
//         std::cout << m_err_stream.str() << std::endl;
//     }
// };

class GenericException : public std::exception {
 protected:
  std::stringstream m_err_stream;

 public:
  GenericException() {}
  explicit GenericException(const std::string& p_err) { m_err_stream << p_err; }

  virtual const char* what() const throw() {
    return m_err_stream.str().c_str();
  }

  virtual void print() const { std::cout << m_err_stream.str() << std::endl; }
};

class InvalidKey : public GenericException {
 public:
  explicit InvalidKey(const std::string& p_key) {
    m_err_stream << p_key << " not found in the metadata\n";
  }

  virtual const char* what() const throw() {
    return m_err_stream.str().c_str();
  }
};

class MemoryAllocationFailure : public GenericException {
 public:
  explicit MemoryAllocationFailure(size_t p_size, bool p_realloc = false) {
    m_err_stream << "Failed to allocate " << p_size << " bytes.";
    if (p_realloc) {
      m_err_stream << "realloc is set to false.\n";
    } else {
      m_err_stream << "\n";
    }
  }
};

class BindFailure : public GenericException {
 public:
  explicit BindFailure(std::string p_err) : GenericException(p_err) {}
};

class NotConnected : public GenericException {
 public:
  explicit NotConnected(std::string p_err) : GenericException(p_err) {}
};

class PacketReceiveFailure : public GenericException {
 public:
  explicit PacketReceiveFailure(std::string p_err) : GenericException(p_err) {}
};

class InvalidSize : public GenericException {
 public:
  explicit InvalidSize(std::string p_msg, size_t p_size) {
    m_err_stream << p_msg << ". Provided size: " << p_size << " \n";
  }

  explicit InvalidSize(size_t p_size) {
    m_err_stream << "Provided size: " << p_size << "\n";
  }
};

// nice utility from here:
// https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
template <typename... Args>
std::string string_format(const std::string format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
               1;  // Extra space for '\0'
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(),
                     buf.get() + size - 1);  // We don't want the '\0' inside
}

#endif  // SRC_EX_EXCEPTIONS_HPP_
