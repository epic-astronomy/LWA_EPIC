---
id: xoj4z82p
title: GulpGen_rft | Raft kernel to generate a gulp
file_version: 1.1.3
app_version: 1.18.21
---

<!--MERMAID {width:100}-->
```mermaid
classDiagram
**\`raft::kernel\`** <|-- GulpGen\_rft : inherits
**class \`raft::kernel\`** {
\+ virtual raft::status run()
}
class GulpGen\_rft{
\-std::unique\_ptr~PktAssmblr~ m\_assmblr
\-int m\_timer

\-...Metrics Variables
+GulpGen\_rft(std::unique\_ptr~PktAssmblr~&, int)
raft::kstatus run ()
}
```
<!--MCONTENT {content: "classDiagram<br/>\n**\\`raft::kernel\\`** <|-- GulpGen\\_rft : inherits<br/>\n**class \\`raft::kernel\\`** {<br/>\n\\+ virtual raft::status run()<br/>\n}<br/>\nclass GulpGen\\_rft{<br/>\n\\-std::unique\\_ptr~PktAssmblr~ m\\_assmblr<br/>\n\\-int m\\_timer\n\n\\-...Metrics Variables<br/>\n+GulpGen\\_rft(std::unique\\_ptr~PktAssmblr~&, int)<br/>\nraft::kstatus run ()<br/>\n}"} --->

<br/>

The `GulpGen_rft`<swm-token data-swm-token=":src/raft_kernels/packet_gen.hpp:44:2:2:`class GulpGen_rft : public raft::kernel {`"/> raft kernel is responsible for fetching a gulp from the packet assembler and send it to the correlator (defined in `CorrelatorRft`<swm-token data-swm-token=":src/raft_kernels/correlator.hpp:42:2:2:`class CorrelatorRft : public raft::kernel {`"/>). It is the starting point of the pipeline. This kernel only defines one output port called `gulp` with a `Payload`<swm-token data-swm-token=":src/ex/buffer.hpp:228:3:3:`  explicit Payload(std::shared_ptr&lt;MBuf&gt; p_mbuf);`"/> type (see [Creating a Raft Kernel](creating-a-raft-kernel.hr4rzvt1.sw.md) for details on port definitions). The payload is a buffer with a `uint8_t` data type that carries 4+4-bit electric field spectra from the F-engine (see [LFbufrMngr | Lock Free Buffer Pools](lfbufrmngr-lock-free-buffer-pools.boxu201d.sw.md) for details on the `Payload`<swm-token data-swm-token=":src/ex/buffer.hpp:211:2:2:`struct Payload {`"/> class). The `GulpGen_rft`<swm-token data-swm-token=":src/raft_kernels/packet_gen.hpp:44:2:2:`class GulpGen_rft : public raft::kernel {`"/> kernel accepts a `unique_ptr`to a `PacketAssembler`<swm-token data-swm-token=":src/ex/packet_assembler.hpp:88:1:1:`  PacketAssembler(std::string p_ip, int p_port, size_t p_nseq_per_gulp = 1000,`"/> instance and assumes it ownership. Gulps are requested using the `get_gulp`<swm-token data-swm-token=":src/ex/packet_assembler.hpp:90:3:3:`  payload_t get_gulp();`"/> function. The user is responsible for properly intializing the packet assembler instance before passing it to the kernel.

<br/>

<!--MERMAID {width:100}-->
```mermaid
flowchart LR
live\_stream["Live Streamer"]
pkt\_gen["Gulp Generator"]
chan\_reducer["Channel
Reducer"]
corr["Correlator
(CUDA-based)"]
lc["Pixel Extractor
(Light curves)"]
idxfetch["Index Fetcher"]
pkt\_gen-->corr
corr-->live\_stream
<br/>corr-->chan\_reducer
chan\_reducer-->idxfetch
idxfetch-->lc
chan\_reducer-->lc
lc-->db["DB Ingester"]
lc-->acc["Accumulator"]
acc-->disksav["Disk Saver"]
style pkt\_gen stroke:red,stroke-width:4px
```
<!--MCONTENT {content: "flowchart LR<br/>\nlive\\_stream\\[\"Live Streamer\"\\]<br/>\npkt\\_gen\\[\"Gulp Generator\"\\]<br/>\nchan\\_reducer\\[\"Channel<br/>\nReducer\"\\]<br/>\ncorr\\[\"Correlator<br/>\n(CUDA-based)\"\\]<br/>\nlc\\[\"Pixel Extractor<br/>\n(Light curves)\"\\]<br/>\nidxfetch\\[\"Index Fetcher\"\\]<br/>\npkt\\_gen\\-\\-\\>corr<br/>\ncorr\\-\\-\\>live\\_stream<br/>\n<br/>corr\\-\\-\\>chan\\_reducer<br/>\nchan\\_reducer\\-\\-\\>idxfetch<br/>\nidxfetch\\-\\-\\>lc<br/>\nchan\\_reducer\\-\\-\\>lc<br/>\nlc\\-\\-\\>db\\[\"DB Ingester\"\\]<br/>\nlc\\-\\-\\>acc\\[\"Accumulator\"\\]<br/>\nacc\\-\\-\\>disksav\\[\"Disk Saver\"\\]<br/>\nstyle pkt\\_gen stroke:red,stroke-width:4px"} --->

<br/>

The kernel can run in timed and untimed modes. The duration for the timed mode is specifed (in seconds) using the `p_timer_s`<swm-token data-swm-token=":src/raft_kernels/packet_gen.hpp:71:15:15:`  GulpGen_rft(std::unique_ptr&lt;PktAssmblr&gt;&amp; p_assmblr, int p_timer_s)`"/> parameter to the constructor. The generator can run continuously by setting the `p_timer_s`<swm-token data-swm-token=":src/raft_kernels/packet_gen.hpp:71:15:15:`  GulpGen_rft(std::unique_ptr&lt;PktAssmblr&gt;&amp; p_assmblr, int p_timer_s)`"/> parameter to -1. Because the dataflow begins in this kernel, we should be able to stop the pipeline by stopping the gulp generation. However, this kernel is yet to respond to kill signals from the OS. So the pipeline cannot be stopped with `Ctrl + C`. It needs to be killed using `Ctrl + Z`.

> _gotcha_: If the program is killed using the `Ctrl + Z` command, it may sometimes result in a zombie process that blocks the metrics end point. This causes the program to segfault upon restarting. So it is advised to use `killall -9 epic++` to properly cleanup the program.

### Further Reading

[CorrelatorRft | Raft Kernel to Image Gulps using EPIC](correlatorrft-raft-kernel-to-image-gulps-using-epic.ppdii2t8.sw.md)

[LFbufrMngr | Lock Free Buffer Pools](lfbufrmngr-lock-free-buffer-pools.boxu201d.sw.md)

[Creating a Raft Kernel](creating-a-raft-kernel.hr4rzvt1.sw.md)

<br/>

This file was generated by Swimm. [Click here to view it in the app](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBTFdBX0VQSUMlM0ElM0FlcGljLWFzdHJvbm9teQ==/docs/xoj4z82p).
