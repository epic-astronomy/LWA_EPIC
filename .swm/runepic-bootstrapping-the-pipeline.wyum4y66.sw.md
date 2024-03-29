---
id: wyum4y66
title: RunEpic | Bootstrapping the pipeline
file_version: 1.1.3
app_version: 1.18.21
---

<!--MERMAID {width:100}-->
```mermaid
classDiagram
EpicKernels<|--EPIC : Inherits

note for EPIC "Intialize kernels\\nBind kernels\\nInitialize pipelines"
class EpicKernels{
\- PktGen\_kt m\_pkt\_gen
\-DummyPktGen\_kt m\_dpkt\_gen
\-EPICCorrelator\_kt m\_correlator
\-ChanReducer\_kt m\_chan\_reducer
\-PixelExtractor\_kt m\_pixel\_extractor
\-IndexFetcher\_kt m\_index\_fetcher
\-DBIngester\_kt m\_db\_ingester
\-Accumulator\_kt m\_accumulator
\-DiskSaver\_kt m\_disk\_saver
\-EpicLiveStream\_kt m\_live\_streamer
+void BindKernels2Cpu(cxxopt::Options)
+void InitMap(cxxopt::Options)
EpicKernels(cxxopt::Options, raft::map\*)
}

class EPIC{

#raft::map m\_map
#EPIC<\_nthGPU-1> m\_next\_epic

EPIC(cxxopt::Options, raft::map\*)

}
```
<!--MCONTENT {content: "classDiagram<br/>\nEpicKernels<|--EPIC : Inherits\n\nnote for EPIC \"Intialize kernels\\\\nBind kernels\\\\nInitialize pipelines\"<br/>\nclass EpicKernels{<br/>\n\\- PktGen\\_kt m\\_pkt\\_gen<br/>\n\\-DummyPktGen\\_kt m\\_dpkt\\_gen<br/>\n\\-EPICCorrelator\\_kt m\\_correlator<br/>\n\\-ChanReducer\\_kt m\\_chan\\_reducer<br/>\n\\-PixelExtractor\\_kt m\\_pixel\\_extractor<br/>\n\\-IndexFetcher\\_kt m\\_index\\_fetcher<br/>\n\\-DBIngester\\_kt m\\_db\\_ingester<br/>\n\\-Accumulator\\_kt m\\_accumulator<br/>\n\\-DiskSaver\\_kt m\\_disk\\_saver<br/>\n\\-EpicLiveStream\\_kt m\\_live\\_streamer<br/>\n+void BindKernels2Cpu(cxxopt::Options)<br/>\n+void InitMap(cxxopt::Options)<br/>\nEpicKernels(cxxopt::Options, raft::map\\*)<br/>\n}\n\nclass EPIC{\n\n#raft::map m\\_map<br/>\n#EPIC<\\_nthGPU-1> m\\_next\\_epic\n\nEPIC(cxxopt::Options, raft::map\\*)\n\n}"} --->

<br/>

The `RunEpic` function is provides the entrypoint for executing the pipeline. It parses command line options and executes the appropriate commands. These options are defined in the `📄 src/ex/option_parser.hpp` and are based on the [cxxopts](https://github.com/jarro2783/cxxopts) library (see [Command Line Options](command-line-options.obfbc4o9.sw.md)). The default command executes the imaging pipeline, and can execute multiple pipelines on separate GPUs by setting the `--ngpus`options to the desired number. Users can alternatively specify the `--printendpoints` command to display the observing frequencies broadcasted on each F-Engine endpoint.

The pipelines are initialized by the `EPIC`<swm-token data-swm-token=":src/raft_kernels/epic_executor.hpp:125:2:2:`class EPIC : public EPICKernels&lt;_nthGPU - 1&gt; {`"/> class and the number of pipelines are set by its template parameter. This class recursively initializes another `EPIC`<swm-token data-swm-token=":src/raft_kernels/epic_executor.hpp:125:2:2:`class EPIC : public EPICKernels&lt;_nthGPU - 1&gt; {`"/> object until the numer of desired pipelines is reached, and are added to a map.

<br/>

Recursive pipeline initialization
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 src/raft_kernels/epic_executor.hpp
```c++
124    template <unsigned int _nthGPU>
125    class EPIC : public EPICKernels<_nthGPU - 1> {
126     protected:
127      raft::map* m_map;
128      EPIC<_nthGPU - 1> m_next_epic;
```

<br/>

The `EPIC`<swm-token data-swm-token=":src/raft_kernels/epic_executor.hpp:125:2:2:`class EPIC : public EPICKernels&lt;_nthGPU - 1&gt; {`"/>class inherits the kernels and pipeline initialization functionality from the `EPICKernels`<swm-token data-swm-token=":src/raft_kernels/epic_executor.hpp:35:2:2:`class EPICKernels {`"/> class, which creates one pipeline at a time. Each member kernel in this class typicallly has a getter function which accepts a `cxxopts::Options`<swm-token data-swm-token=":src/ex/option_parser.hpp:33:0:2:`cxxopts::Options GetEpicOptions() {`"/> object and returns an initialized kernel. These getter functions are defined in the `📄 src/raft_kernels/kernel_types.hpp` file and are used in the member initializer list. After the kernels are initialized, they are bound to separate CPU cores in the `BindKernels2Cpu`<swm-token data-swm-token=":src/raft_kernels/epic_executor.hpp:53:3:3:`  void BindKernels2Cpu(const KernelTypeDefs::opt_t&amp; p_options) {`"/> using `RftManip`<swm-token data-swm-token=":src/ex/helper_traits.hpp:107:2:2:`using RftManip = raft::manip&lt;RftAffinityGrp&lt;AffGrpID&gt;, RftDeviceCpu&lt;CPUID&gt;&gt;;`"/> template. It binbs the kernel based on its two template parameters: `CPUID`, which specifies the CPU the kernel must be bound to and `AffGrpID`, which specifies the kernels affinity. Here the kernels in each pipeline are assigned consecutive core numbers with a gap of 2 between each pipeline. The kernels are linked together appropriately and are added to the raft map in the `InitMap`<swm-token data-swm-token=":src/raft_kernels/epic_executor.hpp:73:3:3:`  void InitMap(const KernelTypeDefs::opt_t&amp; p_options) {`"/> function. Finally, the raft map is executed with all the pipelines.

<br/>

Executing a two pipeline map
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 src/raft_kernels/epic_executor.hpp
```c++
180        auto epic = EPIC<2>(options, &m);
```

<br/>

> _gotcha_: Do not forget to set number of kernels in the `m_nkernels`<swm-token data-swm-token=":src/raft_kernels/epic_executor.hpp:39:9:9:`  static constexpr unsigned int m_nkernels{9};`"/> member. Incorrect values may cause multiple kernels to bind to the same core, which may lead to performance issues and/or undefined behaviour.

### Further reading

[GulpGen_rft | Raft kernel to generate a gulp](gulpgen_rft-raft-kernel-to-generate-a-gulp.xoj4z82p.sw.md)<br/>
[Creating a Raft Kernel](creating-a-raft-kernel.hr4rzvt1.sw.md)<br/>
[Command Line Options](command-line-options.obfbc4o9.sw.md)

<br/>

This file was generated by Swimm. [Click here to view it in the app](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBTFdBX0VQSUMlM0ElM0FlcGljLWFzdHJvbm9teQ==/docs/wyum4y66).
