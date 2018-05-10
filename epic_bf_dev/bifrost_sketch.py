



class epicOp(object):
    def __init__(self, log, voltage_ring, image_ring, nchan=256, nants=256, npol_in=2,
                 npol_out=None, ntime_gulp=2500, guarantee=True, core=-1, gpu=-1):
        """
            Initialize.
            Args:
                log: Log file.
                voltage_ring: Input ring buffer, streaming voltage data from antennas.
                              (on host or device?)
                image_ring: Output ring buffer for images. (host or device?)
                nchan: Number of frequency channels (Default 256)
                nants: Number of antennas (aka stands)
                npol_in: Number of input polarizations (Default 2)
                ntime_gulp:
                guarantee:
                core:
                gpu:
        """



    def main(self):
        cpu_affinity.set_core(self.core)  # TODO: What is this?
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1,
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(), })

        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):  # TODO: what is guarantee?
                # A "sequence" corresponds to a file or observation.
                ihdr = json.loads(iseq.header.tostring())

                # TODO: What does this do?
                self.sequence_proclog.update(ihdr)

                nchan = ihdr['nchan']
                nstand = ihdr['nstand']
                npol = ihdr['npol']

                # TODO: where is configMessage coming from?
                # TODO: configMessage contains gains/delays... do we want to handle differently?
                status = self.updateConfig(self.configMessage(), ihdr, iseq.time_tag,
                                           forceUpdate=True)

                # TODO: generalize data size (e.g. 1 byte for 4+4 complex)
                # Gulp size corresponds to "span" size. A span is a unit of data
                # blocks operate on. Bounded by a single time and full file.
                # A span is made of multiple frames. A frame is a single element
                # of the data stream, corresponding to a single time stamp out of
                # the F-engine.
                igulp_size = self.ntime_gulp_in * nchan * nstand * npol  # 4+4 complex, units of bytes
                # TODO: figure out output size
                # TODO: figure out other output gulps
                ogulp_size = 8 * self.ntime_gulp_out * nchan * self.ngrid * self.npol_out  # complex64, units of bytes

                # TODO: separate stand and pol axis
                ishape = (self.ntime_gulp_in, nchan, nstand * npol)
                oshape = (self.ntime_gulp_out, nchan, self.ngrid, self.npol_out)

                # TODO: recall what this does and document it
                ticksPerTime = int(FS) / int(CHAN_BW)
                base_time_tag = iseq.time_tag

                ohdr = ihdr.copy()
                # TODO: What additional info do we want in the header?
                # Do we need to remove anything that was in input header?
                ohdr['ngrid'] = self.ngrid
                ohdr['nbit'] = 32  # TODO: don't hard code this
                ohdr['complex'] = True  # TODO: Same pol will not be complex... but cross-pol is.
                ohdr_str = json.dumps(ohdr)

                self.oring.resize(ogulp_size)

                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
                    for ispan in iseq.read(igulp_size):
                        if ispan.size < igulp_size:
                            continue  # Ignore partial gulp
                        curr_time = time.time()
                        acquire_time = curr_time - prev_time
                        prev_time = curr_time

                        with oseq.reserve(ogulp_size) as ospan:
                            curr_time = time.time()
                            reserve_time = curr_time - prev_time
                            prev_time = curr_time

                            # Setup and load
                            idata = ispan.data_view(np.uint8).reshape(ishape)
                            # TODO: What does odata contain at this point? Nothing?
                            odata = ospan.data_view(np.complex64).reshape(oshape)

                            # Fix the type
                            # TODO: Understand this better
                            bfidata = BFArray(shape=idata.shape, dtype='ci4', native=False,
                                              buffer=idata.ctypes.data)

                            # Copy to cuda memory (see tdata in init)
                            copy_array(self.tdata, bfidata)

                            # TODO: Our pipeline!!!
                            # TODO: Look at self.bfbf.matmul for example - what is it doing? How does it handle time?
                            # TODO: apply cal
                            # TODO: Grid
                            # TODO: 2DFFT
                            # TODO: Can we process multiple at once?
                            # TODO: gen cal
                            # TODO: square/cross product, average
                            # TODO: remove auto -- is this done in GPU or offline? do we need more info to do it offline?

                            # Save and cleanup
                            # TODO: "bdata" will probably be someting else
                            odata[...] = self.bdata.copy(space='system').transpose(X)

                        # Update the base time tag
                        base_time_tag += self.ntime_gulp * ticksPerTime

                        # Check for an update to the configuration
                        self.updateConfig(self.configMessage(), ihdr, base_time_tag, forceUpdate=False)

                        # TODO: Are these time keepings actually used?
                        curr_time = time.time()
                        process_time = curr_time - prev_time
                        prev_time = curr_time
                        self.perf_proclog.update({'acquire_time': acquire_time,
                                                  'reserve_time': reserve_time,
                                                  'process_time': process_time, })
