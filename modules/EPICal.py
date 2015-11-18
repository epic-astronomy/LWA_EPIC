import numpy as NP
import ipdb as PDB
import scipy.constants as FCNST

       
class cal:
    """
    -------------------------------------------------------------------------------
    Class to handle everything calibration for EPIC
  
    *** Attributes ***

    n_ant:              [integer] Number of antennas. Calculated from ant_pos.

    n_chan:             [integer] Number of frequenc channels. Calculated from freqs.

    freqs:              Numpy array with frequencies in Hz.

    ant_pos:            Numpy array with antenna locations in wavelengths for each frequency.
                        Dimensions: n_ant x n_chan x 3. 
                        If n_ant x 3 supplied, assumed to be in meters and rebroadcasted in init.

    ref_ant:            [integer] Antenna to fix phase. Default = 0.

    freq_ave:           [integer] Number of frequencies to average when solving for gains.

    pol:                Polarization key (to reference image object). Default = 'P1'.

    curr_gains:         Complex numpy array representing current estimate of gains.
                        Typically not passed in, Default = ones.
                        Dimension: n_ant x n_chan

    cal_corr:           Complex numpy array, correlation currently being integrated/averaged.
                        Dimension: n_ant x n_chan

    sim_mode:           [Boolean] represting whether in simulation mode. Default = False.

    sim_gains:          Complex numpy array representing the simulated gains (if sim_mode),
                        which are applied to the simulated data.

    n_iter:             [integer] Number of iterations before updating the calibration.
                        Default = 10.

    count:              [integer] Current iteration number.

    gain_factor:        [float] Weighting factor when updating gains. Default = 10.

    inv_gains:          [Boolean] If False (default), calibrate by dividing by gains. If
                        True, calibrate by multiplying by conjugate of gains.

    phase_fit:          [Boolean] Whether to fit only gain phase.
                        Default = False

    sky_model:          Numpy array containing (l,m,n,flux) for sources in sky model for each
                        channel. Default = 1 Jy source on center.
                        Dimensions: n_source x n_chan x 4

    auto_noise_model:   [float or numpy float array] Modeled noise level for auto visibilities.
                        Default = 0.

    exclude_autos:      [Boolean] Whether to exclude autos from calibration solutions.
                        Default = False

    auto_corr:          Numpy array, auto correlations. Used if exclude_autos.
                        Dimension = n_ant x n_chan

    model_vis:          Model visibilities. Calculated in cal loop.
                        Dimensions: n_ant x n_ant x n_chan

    cal_source:         [integer] Number pointing to calibration source (indexing the first
                        dimension of sky_model). Default = brightest in model.
                        TODO: make default the brightest _beam weighted_ source in model.

    cal_pix_loc:        Location (l,m,n) of pixel closest to cal_source. Calculated in loop.

    cal_pix_ind:        Index (x,y) of pixel closest to cal_source. Calculated in loop.

    *** Functions ***

    simulate_gains:     Create a set of gains for simulation purposes.

    scramble_gains:     Apply a random jitter to the current gains. Useful for testing/simulating.

    update_model_vis:   Updates the model visibilities and cal pixel.

    update_cal:         Updates the calibration solution given the curr_gains, sky_model,
                        and input data.

    calc_corr:           Calculate correlation needed for calibration.

    apply_cal:          Applies current gain solutions to the data.

    ----------------------------------------------------------------------------------
    """

    def __init__(self, freqs, ant_pos, ref_ant=0, freq_ave=1, pol='P1', curr_gains=None, sim_mode=False, 
        n_iter=10, gain_factor=1.0, inv_gains=False, sky_model=NP.ones(1,dtype=NP.float32), cal_source=None, 
        phase_fit=False, auto_noise_model=0.0, exclude_autos=False):

        # Get derived values and check types, etc.
        n_chan = freqs.shape[0]
        if not ant_pos.shape[-1] == 3:
            raise ValueError('Antenna positions must be three dimensional!')
        n_ant = ant_pos.shape[0]
        if len(ant_pos.shape) != 3:
            if len(ant_pos.shape) == 2:
                print 'Antenna positions assumed to be in meters. Reformatting to wavelengths.'
                ant_pos = ant_pos.reshape(n_ant,1,3)
                ant_pos = ant_pos * freqs.reshape(1,n_chan,1)/FCNST.c
            else:
                ValueError('Antenna positions in wrong format!')
        elif ant_pos.shape[1] != n_chan:
            ValueError('Antenna positions do not match frequencies!')

        if (gain_factor > 1.0) or (gain_factor < 0.0): gain_factor = 1.0

        # Defaults for sky_mdoel:
        #   If no spectral info given, assume constant
        #   If no position given, and only one source, assume on center.
        if len(sky_model.shape) == 1:
            if sky_model.shape[0] == 1:
                # Assume just given flux.
                temp_model = NP.zeros((1,n_chan,4))
                temp_model[0,:,3] = sky_model
                temp_model[0,:,2] = 1
                sky_model = temp_model
            elif sky_model.shape[0] == 4:
                # Assume given position, flux correctly.
                sky_model = NP.tile(sky_model,(1,n_chan,1))
            elif sky_model.shape[0] == 3:
                # Assume position, flux is given by (l,m,flux) (missing n)
                temp_model = NP.zeros(4)
                temp_model[0:2] = sky_model[0,:]
                temp_model[2] = NP.sqrt(1-NP.sum(temp_model[0:2]**2))
                temp_model[3] = sky_model[2]
                sky_model = NP.tile(temp_model,(1,n_chan,1))
            else:
                ValueError('Unrecognized sky model format!')
        elif len(sky_model.shape) != 3:
            # TODO: more cases. But this should be ok for now.
            ValueError('Unrecognized sky model format!')



        # Assign attributes
        self.n_ant = n_ant
        self.n_chan = n_chan
        self.freqs = freqs
        self.ant_pos = ant_pos
        self.ref_ant = ref_ant
        self.freq_ave = freq_ave
        self.pol = pol
        if curr_gains is None:
            self.curr_gains = NP.ones((n_ant,n_chan), dtype=NP.complex64)
        else:
            self.curr_gains = curr_gains
        self.cal_corr = NP.zeros((n_ant,n_chan), dtype=NP.complex64)
        self.sim_mode = sim_mode
        if sim_mode:
            self.sim_gains = self.simulate_gains()
        self.n_iter = n_iter
        self.count = 0
        self.gain_factor = gain_factor
        self.inv_gains = inv_gains
        self.phase_fit = phase_fit
        self.sky_model = sky_model
        self.auto_noise_model = auto_noise_model
        self.exclude_autos = exclude_autos
        self.auto_corr = NP.zeros((n_ant,n_chan), dtype=NP.float32)
        if cal_source is None:
            # Use brightest source in model
            self.cal_source = sky_model[:,n_chan/2,3].argmax()
        else:
            self.cal_source = cal_source
        self.cal_pix_ind = None # placeholder until it can be calculated.
        # model_vis, cal_pix_loc, and cal_pix_ind are determined after an imgobj is passed in.

    ####################################

    def simulate_gains(self, scale=0.25):
        # Assign some random complex numbers

        ang = NP.random.uniform(low=0.0, high=2*NP.pi, size=[self.n_ant,self.n_chan])
        amp = NP.abs(NP.random.normal(loc=1,scale=scale,size=[self.n_ant,self.n_chan]))

        return amp*NP.exp(1j*ang)

    ######

    def scramble_gains(self, amp):
        # Mix up gains a bit
        self.curr_gains += NP.random.normal(0,NP.sqrt(amp),self.curr_gains.shape) + 1j * NP.random.normal(0,NP.sqrt(amp),self.curr_gains.shape)

        return

    ######

    def update_model_vis(self, gridl, gridm):
        # This function will generate the model visibilities for calibration.
        # Eventually this should be moved to Nithya's modules to account for beams,
        # but for testing we'll do it here.

        # First find the appropriate pixel to phase to.
        xind,yind = NP.unravel_index(NP.argmin((gridl-self.sky_model[self.cal_source,0,0])**2+(gridm-self.sky_model[self.cal_source,0,1])**2),gridl.shape) 
        self.cal_pix_ind = NP.array([xind,yind])
        self.cal_pix_loc = NP.array([gridl[xind,yind], gridm[xind,yind], NP.sqrt(1-gridl[xind,yind]**2-gridm[xind,yind]**2)])

        # Reshape arrays to match: n_src x n_ant x n_chan x 3
        n_src = self.sky_model.shape[0]
        model_pos = NP.reshape(self.sky_model[:,:,0:3],(n_src,1,self.n_chan,3))
        model_flux = NP.reshape(self.sky_model[:,:,3], (n_src,1,self.n_chan))
        ant_pos = NP.reshape(self.ant_pos, (1,self.n_ant,self.n_chan,3))

        # Calculate model visibilities
        # TODO: Incorporate beam
        model_vis = NP.zeros((self.n_ant, self.n_ant, self.n_chan), dtype=NP.complex128)
        for ant in xrange(self.n_ant):
            model_vis[ant,:,:] = NP.sum(model_flux * NP.exp(-1j * 2*NP.pi * NP.sum(model_pos * (ant_pos-NP.reshape(ant_pos[:,ant,:,:],(1,1,self.n_chan,3))),axis=3)),axis=0)

        # Add auto noise term
        if isinstance(self.auto_noise_model,NP.float):
            # Just given one number, make it a diagonal matrix
            self.auto_noise_model = self.auto_noise_model * NP.ones(self.n_ant)
        if len(self.auto_noise_model) == self.n_ant:
            self.auto_noise_model = NP.diag(self.auto_noise_model).reshape((self.n_ant,self.n_ant,1))

        model_vis = model_vis + self.auto_noise_model

        # Remove auto term if excluding
        if self.exclude_autos:
            for ant in xrange(self.n_ant):
                model_vis[ant,ant,:] = 0.0

        self.model_vis = model_vis

    ######

    def update_cal(self, Edata, imgobj):
        # Check if correlation pixel is known
        if self.cal_pix_ind is None:
            self.update_model_vis(imgobj.gridl, imgobj.gridm)

        imgdata = imgobj.holimg[self.pol][self.cal_pix_ind[0],self.cal_pix_ind[1],:].flatten()

        self.calc_corr(Edata,imgdata)

        if self.count == self.n_iter:
            # Reached integration level, update the estimated gains
            print NP.mean(NP.abs(self.cal_corr))
            # Expression depends on type of calibration
            if self.inv_gains:
                # Inverted gains version
                temp_gains = self.n_ant * self.cal_corr / NP.sum(self.n_iter * NP.exp(1j * 2*NP.pi * NP.sum(self.ant_pos * self.cal_pix_loc.reshape(1,1,3),axis=2)) * self.model_vis * NP.reshape(NP.abs(self.curr_gains)**2,(1,self.n_ant,self.n_chan)), axis=1)
            else:
                # Regular version
                temp_gains = self.n_ant * self.cal_corr / NP.sum(self.n_iter * NP.exp(1j * 2*NP.pi * NP.sum(self.ant_pos * self.cal_pix_loc.reshape(1,1,3),axis=2)) * self.model_vis, axis=1)
                
            # TODO:
            # Account for beam value at cal source location!
            
            # Average in frequency
            for i in NP.arange(NP.ceil(NP.float(self.n_chan)/self.freq_ave)):
                mini=i*self.freq_ave
                maxi=NP.min((self.n_chan,(i+1)*self.freq_ave))
                temp_gains[:,mini:maxi] = NP.mean(temp_gains[:,mini:maxi],axis=1).reshape(self.n_ant,1)

            # Fix ref_ant's phase.
            phasor = temp_gains[self.ref_ant,:]/NP.abs(temp_gains[self.ref_ant,:])
            temp_gains = temp_gains * NP.conj(phasor).reshape(1,self.n_chan)
            if self.phase_fit:
                # Only fit phase
                temp_gains = temp_gains / NP.abs(temp_gains)

            self.curr_gains = self.curr_gains * (1 - self.gain_factor) + self.gain_factor * temp_gains
            if self.phase_fit:
                # phase correction with gain_factor could result in small amplitude drop. Fix this.
                self.curr_gains = self.curr_gains / NP.abs(self.curr_gains)

            self.count = 0
            self.cal_corr = NP.zeros((self.n_ant,self.n_chan), dtype=NP.complex64)

    ######

    def calc_corr(self, Edata, imgdata):
        # Perform the correlation of antenna data with image output.
        # Kind of silly to separate this into a function, but conceptually it makes sense.
        self.cal_corr = self.cal_corr + Edata*NP.reshape(NP.conj(imgdata),(1,self.n_chan))
        self.count += 1

    ######

    def apply_cal(self, data, meas=False):
        # Function to apply calibration solution to data.

        if self.curr_gains.size != data.size:
            raise ValueError('Data does not match calibration gain size')

        if meas:
            # In measurement mode, apply sim_gains.
            data = data * self.sim_gains
        else:
            # In calibration mode.
            if self.inv_gains:
                # apply inverse gains - i.e. multiply by gain to calibrate
                data = data * NP.conj(self.curr_gains)
            else:
                data = data / self.curr_gains

        return data


