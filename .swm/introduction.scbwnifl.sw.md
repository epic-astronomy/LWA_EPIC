---
id: scbwnifl
title: Introduction
file_version: 1.1.3
app_version: 1.18.21
---

The EPIC Imager is an all-sky radio imaging system built for realtime operations. EPIC stands for E-field Parallel Imaging Correlator algorithm ([Thyagarajan et al. 2017]([https://academic.oup.com/mnras/article/467/1/715/2917985](https://academic.oup.com/mnras/article/467/1/715/2917985))), which transforms the observed electric fields directly into sky-images without calculating _visibilities_. This direct transformation makes EPIC computationally far less intensive compared to regular imaging. The imager code is mainly written in C++17 with occasional calls to python modules. The pipeline takes the data from the F-engine through several transformative stages to produce multiple data products including FITS images, light curves and live video streams. Each stage in the pipeline, for example, accumulation, is implemented as a kernel using the [Raftlib C++](https://github.com/RaftLib/RaftLib) library, although the actual correlator code is implemented using [CUDA](https://developer.nvidia.com/cuda-toolkit). Raft provides lock-free queues to efficiently transfer data between kernels. Any non-primitive data types, for instance, images or E-field spectra, are transferred using lock-free buffers pools. The flowchart below shows the data flow in the EPIC imaging pipeline along with its external components.

<br/>

<!--MERMAID {width:100}-->
```mermaid
flowchart TB
live\_stream["Live Streamer"]
rtmp["RTMP Server"]
epictv["EPIC TV"]
idxfetch["Index Fetcher"]
lc["Pixel Extractor
(Light curves)"]
subgraph ip ["Imaging Pipeline"]
direction LR
pkt\_gen["Gulp Generator"]
chan\_reducer["Channel
Reducer"]
corr["Correlator
(CUDA-based)"]
<br/>pkt\_gen-->corr
corr-->live\_stream
<br/>corr-->chan\_reducer
chan\_reducer-->idxfetch
idxfetch-->lc
chan\_reducer-->lc
lc-->db["DB Ingester"]
lc-->acc["Accumulator"]
acc-->disksav["Disk Saver"]
end
F-Engine-->ip
live\_stream-->rtmp
rtmp-->epictv
db-->pg["Postgres DB"]
disksav-->Disk
Watchdog-->idxfetch
ev["Events
(CHIME, Realfast, DSA-110...)"]-->Watchdog

```
<!--MCONTENT {content: "flowchart TB<br/>\nlive\\_stream\\[\"Live Streamer\"\\]<br/>\nrtmp\\[\"RTMP Server\"\\]<br/>\nepictv\\[\"EPIC TV\"\\]<br/>\nidxfetch\\[\"Index Fetcher\"\\]<br/>\nlc\\[\"Pixel Extractor<br/>\n(Light curves)\"\\]<br/>\nsubgraph ip \\[\"Imaging Pipeline\"\\]<br/>\ndirection LR<br/>\npkt\\_gen\\[\"Gulp Generator\"\\]<br/>\nchan\\_reducer\\[\"Channel<br/>\nReducer\"\\]<br/>\ncorr\\[\"Correlator<br/>\n(CUDA-based)\"\\]<br/>\n<br/>pkt\\_gen\\-\\-\\>corr<br/>\ncorr\\-\\-\\>live\\_stream<br/>\n<br/>corr\\-\\-\\>chan\\_reducer<br/>\nchan\\_reducer\\-\\-\\>idxfetch<br/>\nidxfetch\\-\\-\\>lc<br/>\nchan\\_reducer\\-\\-\\>lc<br/>\nlc\\-\\-\\>db\\[\"DB Ingester\"\\]<br/>\nlc\\-\\-\\>acc\\[\"Accumulator\"\\]<br/>\nacc\\-\\-\\>disksav\\[\"Disk Saver\"\\]<br/>\nend<br/>\nF-Engine\\-\\-\\>ip<br/>\nlive\\_stream\\-\\-\\>rtmp<br/>\nrtmp\\-\\-\\>epictv<br/>\ndb\\-\\-\\>pg\\[\"Postgres DB\"\\]<br/>\ndisksav\\-\\-\\>Disk<br/>\nWatchdog\\-\\-\\>idxfetch<br/>\nev\\[\"Events<br/>\n(CHIME, Realfast, DSA-110...)\"\\]\\-\\-\\>Watchdog<br/>\n<br/>"} --->

<br/>

Packets from the F-Engine are buffered for a specified duration and grouped into _gulps_, and are sent to the GPU for imaging. The images are transferred to the live stream, and are also further reduced for disk saving and light curve extraction purposes. The imager also periodically fetches sources currently in the field of view from the watchdog to extract their light curves. The watchdog keeps track of all the sources of interest (pulsars, FRBs etcetera).

## Implementation Overview

The main entrypoint into the imaging pipeline is through the `RunEpic`<swm-token data-swm-token=":src/raft_kernels/epic_executor.hpp:144:2:2:`void RunEpic(int argc, char** argv) {`"/> function. It initializes an `EPIC`<swm-token data-swm-token=":src/raft_kernels/epic_executor.hpp:125:2:2:`class EPIC : public EPICKernels&lt;_nthGPU - 1&gt; {`"/>object which recursively creates instances of `EPICKernels`<swm-token data-swm-token=":src/raft_kernels/epic_executor.hpp:35:2:2:`class EPICKernels {`"/> class. Each instance (total number set by `--ngpus` option) runs a separate pipeline on a dedicated GPU. We use one RTX 4090 GPU per node for operations at the LWA Sevilleta station in New Mexico. To add a new kernel to the pipeline, create a kernel member in the `EPICKernels`<swm-token data-swm-token=":src/raft_kernels/epic_executor.hpp:35:2:2:`class EPICKernels {`"/> class and initialize it in the constructor.

<br/>

Adding the kernel object as a private member. The suffix `_kt` indicates a kernel type.
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 src/raft_kernels/epic_executor.hpp
```c++
35     class EPICKernels {
36      private:
37       std::unique_ptr<Streamer> m_streamer;
38       bool m_is_offline{false};
39       static constexpr unsigned int m_nkernels{9};
40       PktGen_kt m_pkt_gen;
41       DummyPktGen_kt m_dpkt_gen;
42       EPICCorrelator_kt m_correlator;
```

<br/>

After initialization, each kernel, which is run on a separate thread, is bound to a dedicated core before adding it to the Raft `m_map`<swm-token data-swm-token=":src/raft_kernels/epic_executor.hpp:52:6:6:`  raft::map* m_map;`"/>. This map defines a directed acyclic graph for executing the pipeline.

<br/>

Binding kernel to a core
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 src/raft_kernels/epic_executor.hpp
```c++
53       void BindKernels2Cpu(const KernelTypeDefs::opt_t& p_options) {
54         // use an additional offset of 2 for the runtime
55         constexpr unsigned int cpu_ofst = (_GpuId)*m_nkernels + 2;
56         // ensure the cpu ID is always non-zero.
57         // Setting it to zero causes instability
58         if (m_is_offline) {
59           RftManip<1 + cpu_ofst, 1 + _GpuId>::bind(m_dpkt_gen);
60         } else {
61           RftManip<1 + cpu_ofst, 1 + _GpuId>::bind(*(m_pkt_gen.get()));
62         }
63         RftManip<2 + cpu_ofst, 1 + _GpuId>::bind(m_correlator);
```

<br/>

Kernels can be joined using the `>>` operator. See [Create a Raft Kernel](create-a-raft-kernel.hr4rzvt1.sw.md) for details. The code below shows the initial part of the pipeline buildup where the data flows from packet assembler to the pixel extractor.

<br/>

Adding a kernel to the pipeline
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 src/raft_kernels/epic_executor.hpp
```c++
73       void InitMap(const KernelTypeDefs::opt_t& p_options) {
74         auto& m = *m_map;
75         // m += m_dpkt_gen >> m_correlator >> m_disk_saver;
76         if (m_is_offline) {
77           m += m_dpkt_gen >> m_correlator["gulp"]["img"] >>
78                m_chan_reducer["in_img"]["out_img"] >> m_pixel_extractor["in_img"];
79         } else {
80           m += *(m_pkt_gen.get()) >> m_correlator["gulp"]["img"] >>
81                m_chan_reducer["in_img"]["out_img"] >> m_pixel_extractor["in_img"];
82         }
83         m += m_correlator["img_stream"] >> m_live_streamer["in_img"];
```

<br/>

Values in the brackets indicate the input and output ports of the kernels. Each kernel defines input and output ports to communicate with other kernels. For example, the correlator kernel defines one input port `gulp` that receives data from the packet assembler. It also defines two output ports `img` and `img_stream`that transmits images to the channel reducer and the live streamer, respectively.

All the kernel types are defined in the `📄 src/raft_kernels/kernel_types.hpp` file. Types defined using the `KernelTypeDefs`<swm-token data-swm-token=":src/raft_kernels/kernel_types.hpp:63:2:2:`struct KernelTypeDefs {`"/> struct, and we specialize a `get_kernel`<swm-token data-swm-token=":src/raft_kernels/kernel_types.hpp:82:3:3:`  ktype get_kernel();`"/> template function to return a kernel object. For instance, the following snippet shows the definition of `EpicLiveStream_kt`<swm-token data-swm-token=":src/raft_kernels/kernel_types.hpp:190:2:2:`using EpicLiveStream_kt = Kernel&lt;_LIVE_STREAMER&gt;::ktype;`"/>, which is an alias for the `EpicLiveStream`<swm-token data-swm-token=":src/raft_kernels/epic_live_streamer.hpp:40:2:2:`class EpicLiveStream : public raft::kernel {`"/> kernel, and its getter function.

<br/>

Defining a kernel type for `EPICKernels`<swm-token data-swm-token=":src/raft_kernels/epic_executor.hpp:35:2:2:`class EPICKernels {`"/> class
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 src/raft_kernels/kernel_types.hpp
```c++
181    template <>
182    struct Kernel<_LIVE_STREAMER> : KernelTypeDefs {
183      using ktype = EpicLiveStream<payload_float_t>;
184      template <unsigned int _GpuId>
185      static ktype get_kernel(const opt_t& options) {
186        return ktype();
187      }
188    };
189    
190    using EpicLiveStream_kt = Kernel<_LIVE_STREAMER>::ktype;
```

<br/>

The `_GpuId`is a template parameter that indicates the ID of the GPU device that executes the pipeline.

The Imager code also uses python modules for a few operations including calculating antenna positions and phases, generating gridding kernels, talking to the ADP, among others. Although the same operations can be performed in C++, using python simplifies coding for these one-time calculations using libraries such as `scipy`without re-compiling the code. The imager uses [pybind11](https://github.com/pybind/pybind11) library to invoke python functions from the C++ code. The code below shows how to fetch ADP start time from the unix epoch by calling a python function.

<br/>

Calling a python function from C++ using pybind11
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 src/ex/py_funcs.hpp
```c++
326    double GetAdpTimeFromUnixEpoch() {
327      py::gil_scoped_acquire acquire;
328      return py::module_::import("epic_utils")
329          .attr("get_ADP_time_from_unix_epoch")()
330          .cast<double>();
331    }
```

<br/>

### Code Organization

The raft kernel definitions are located in `📄 src/raft_kernels` folder and their related definitions in `📄 src/ex` folder. Because the majority of the code is templatized, classes are declared and defined in `hpp` files in the `📄 src/ex` folder. Python modules are located in `📄 src/python` and their C++ counterparts in the `📄 src/ex/py_funcs.hpp` file. Shell scripts to start the imaging pipline and overclock the GPU are in the `📄 src/commands` folder. All external dependencies like `glog` and `pqxx` are included as submodules to the imaging project and are placed in the `📄 src/extern` folder. Finally, the pipeline is built using `CMake`. The CPU and GPU codes are compiled separately and are the linked to the executable. See `📄 CMakeLists.txt` for details.

## Further Reading

**Code Walkthroughs**: Documents in the `code walkthroughs`folder provides implementation details on all the data flows that happen in the imager. Where necessary they also provide tips on extending the code.

**Tutorials:** The `tutorials` folder provides tutorials on adding new features to the code, for instance, building new kernels. It also provides details on debugging the code and lists several _gotchas_ that previously led to severe bugs.

<br/>

This file was generated by Swimm. [Click here to view it in the app](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBTFdBX0VQSUMlM0ElM0FlcGljLWFzdHJvbm9teQ==/docs/scbwnifl).
