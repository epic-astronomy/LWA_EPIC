



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
                ihdr = json.loads(iseq.header.tostring())

                self.sequence_proclog.update(ihdr)

                nchan = ihdr['nchan']
                nstand = ihdr['nstand']
                npol = ihdr['npol']

                # TODO: where is configMessage coming from?
                # TODO: configMessage contains gains/delays... do we want to handle differently?
                status = self.updateConfig(self.configMessage(), ihdr, iseq.time_tag,
                                           forceUpdate=True)

                # TODO: generalize data size (e.g. 1 byte for 4+4 complex)
                igulp_size = self.ntime_gulp_in * nchan * nstand * npol  # 4+4 complex, units of bytes
                # TODO: figure out output size
                # TODO: figure out other output gulps
                ogulp_size = 8 * self.ntime_gulp_out * nchan * self.ngrid * self.npol_out  # complex64, units of bytes

                # TODO: separate stand and pol axis
                ishape = (self.ntime_gulp_in, nchan, nstand * npol)
                oshape = (self.ntime_gulp_out, nchan, self.ngrid, self.npol_out)
