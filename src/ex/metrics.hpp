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

#ifndef SRC_EX_METRICS_HPP_
#define SRC_EX_METRICS_HPP_

#include <prometheus/counter.h>
#include <prometheus/exposer.h>
#include <prometheus/registry.h>
#include <prometheus/summary.h>
#include <prometheus/info.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <sstream>
#include <type_traits>
#include <vector>

namespace pm = prometheus;

/**
 * @brief Prometheus metrics collection interface for EPIC. Initialize
 *
 */
class PrometheusExporter {
  using guage_family_t = pm::Family<pm::Gauge>;
  using counter_family_t = pm::Family<pm::Gauge>;
  using summary_family_t = pm::Family<pm::Summary>;
  using info_family_t = pm::Family<pm::Info>;

 private:
  inline static std::unique_ptr<PrometheusExporter> m_instance{nullptr};
  inline static std::unique_ptr<pm::Exposer> m_exposer{nullptr};
  inline static std::shared_ptr<pm::Registry> m_registry{nullptr};
  inline static guage_family_t *m_runtime_gauge_family_ptr{nullptr};
  inline static std::vector<pm::Gauge *> m_runtime_gauges;
  inline static summary_family_t *m_runtime_summary_family_ptr{nullptr};
  inline static info_family_t *m_runtime_info_family_ptr{nullptr};
  inline static std::vector<pm::Summary *> m_runtime_summaries{nullptr};
  // counter_family_t *m_metric_counter_ptr{nullptr};
  std::string m_bind_addr;
  inline static std::mutex pmreg_mutex;
  inline static bool m_disable_collection{false};

  inline static pm::Summary::Quantiles default_quantiles;
  inline static std::chrono::milliseconds default_summary_window;

  // gauge
  std::string m_runtime_gauge_name = "epic_performance_gauges";
  std::string m_runtime_gauge_help =
      "Performance metrics for individual kernels in epic";
  std::string m_runtime_summary_name = "epic_performance_summary";
  std::string m_runtime_summary_help =
      "Streaming summary metrics for individual kernels in epic";
  std::string m_runtime_info_name = "epic_runtime_info";
  std::string m_runtime_info_help = 
      "EPIC runtime options";

  explicit PrometheusExporter(const std::string &p_bind_addr,
                              bool p_disable_coll)
      : m_bind_addr(p_bind_addr) {
    m_disable_collection = p_disable_coll;
    default_quantiles = {{0.5, 0.05}, {0.90, 0.01}, {0.99, 0.001}};
    default_summary_window = std::chrono::seconds{5};
    m_exposer = std::make_unique<pm::Exposer>(p_bind_addr);
    m_registry = std::make_shared<pm::Registry>();
    m_exposer.get()->RegisterCollectable(m_registry);

    // create a guage for kernel runtimes
    auto &runtime_gauge = pm::BuildGauge()
                              .Name(m_runtime_gauge_name)
                              .Help(m_runtime_gauge_help)
                              .Register(*m_registry);
    m_runtime_gauge_family_ptr = &runtime_gauge;

    auto &runtime_summary = pm::BuildSummary()
                                .Name(m_runtime_summary_name)
                                .Help(m_runtime_summary_help)
                                .Register(*m_registry);
    
    m_runtime_summary_family_ptr = &runtime_summary;

    auto &runtime_info = pm::BuildInfo()
                              .Name(m_runtime_info_name)
                              .Help(m_runtime_info_help)
                              .Register(*m_registry);
    
    m_runtime_info_family_ptr = &runtime_info;
  }

 public:
  PrometheusExporter(const PrometheusExporter &other) = delete;
  void operator=(const PrometheusExporter &) = delete;

  /**
   * @brief Get the PrometheusExporter Instance
   *
   * @param bind_addr Address to expose the metrics. Must include the port.
   * Defaults to 127.0.0.1:8080.
   * @return Returns a pointer to the PrometheusExporter object
   */
  static PrometheusExporter *GetInstance(std::string bind_addr = "",
                                         bool disable_coll = false) {
    if (!m_instance) {
      m_instance = std::unique_ptr<PrometheusExporter>(
          new PrometheusExporter(bind_addr, disable_coll));
    }
    return m_instance.get();
  }

  /**
   * @brief Add a label to the run time gauge
   *
   * @param p_labels Lables for the gauge. Can be speicifed with initializer
   * lists. E.g., {{"key","value"}}
   * @return ID for the provided label
   */
  static unsigned int AddRunTimeGaugeLabel(const pm::Labels &p_labels) {
    if (m_disable_collection) return 0;
    assert(m_instance && "Metrics interface is uninitialized");
    std::lock_guard<std::mutex> lock(pmreg_mutex);
    m_runtime_gauges.push_back(&(m_runtime_gauge_family_ptr->Add(p_labels)));
    return m_runtime_gauges.size() - 1;
  }

  /**
   * @brief Set the Run Time gauge value
   *
   * @param p_id ID for the label to be set
   * @param p_value Value to set the gauge
   */
  static void SetRunTimeGauge(unsigned int p_id, double p_value) {
    if (m_disable_collection) return;
    m_runtime_gauges[p_id]->Set(p_value);
  }

  /**
   * @brief Add a runtime summary label to the registry
   * 
   * @param p_labels Labels
   * @return Returns the registry id for the specified label
   */
  static unsigned int AddRuntimeSummaryLabel(const pm::Labels &p_labels) {
    if (m_disable_collection) return 0;
    assert(m_instance && "Metrics interface is uninitialized");
    std::lock_guard<std::mutex> lock(pmreg_mutex);
    m_runtime_summaries.push_back(&(m_runtime_summary_family_ptr->Add(
        p_labels, default_quantiles, default_summary_window)));
    
    return m_runtime_summaries.size()-1;
  }

  /**
   * @brief Observe the runtime value for the specified summary
   * 
   * @param p_id Registry ID for the summary
   * @param p_value Value to observe
   */
  static void ObserveRunTimeValue(unsigned int p_id, double p_value){
    if(m_disable_collection) return;
    if(p_id>= m_runtime_summaries.size()){
      assert("Invalid summary id");
    }
    m_runtime_summaries[p_id]->Observe(p_value);
  }

  /**
   * @brief Add labels to the info metric
   * 
   * @param p_labels Labels to add
   */
  static void AddInfoLabels(const pm::Labels &p_labels){
    m_runtime_info_family_ptr->Add(p_labels);
  }

  // static std::shared_ptr<pm::Registry> GetRegistry() {
  //   assert(!m_instance && "Metrics interface is uninitialized");
  //   std::lock_guard<std::mutex> lock(pmreg_mutex);
  //   m_registries.push_back(std::make_shared<pm::Registry>());
  //   m_exposer.RegisterCollectible(m_registries.back());
  //   return m_registries.back();
  // }
};

/**
 * @brief Simple function execution timer
 *
 */
class Timer {
  using ClockT = std::chrono::high_resolution_clock;
  using timep_t = typename ClockT::time_point;
  timep_t _start = ClockT::now(), _end = {};

 public:
  void Tick() {
    _end = timep_t{};
    _start = ClockT::now();
  }

  void Tock() { _end = ClockT::now(); }

  double Duration() const {
    assert(_end != timep_t{} && "toc before reporting");
    return std::chrono::duration<double>(_end - _start).count();
  }
};



#endif  // SRC_EX_METRICS_HPP_
