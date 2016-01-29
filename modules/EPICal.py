import numpy as NP
import ipdb as PDB
import scipy.constants as FCNST
import progressbar as PGB

       
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
                        Dimension: n_cal_sources x n_ant x n_chan

    sim_mode:           [Boolean] represting whether in simulation mode. Default = False.

    sim_gains:          Complex numpy array representing the simulated gains (if sim_mode),
                        which are applied to the simulated data.

    n_iter:             [integer] Number of iterations before updating the calibration.
                        Default = 10.

    count:              [integer] Current iteration number.

    ant_twt:            Integer array storing the time weights for each antenna.
                        Dimension = n_ant x n_chan

    damping_factor:     [float] Dampening factor when updating gains. Default = 0.

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

    cal_sources:        [numpy array] Number pointing to calibration source(s) (indexing 
                        the first dimension of sky_model). Default = brightest in model.
                        Dimensions: n_cal_sources (numpy array even if only 1 source)
                        TODO: make default the brightest _beam weighted_ source in model.

    n_cal_sources:      [integer] Number of sources to calibrate with. If cal_sources is supplied,
                        n_cal_sources will be overwritten by dimension of cal_sources.
                        Default = 1

    cal_pix_locs:       Location (l,m,n) of pixels closest to cal_sources. Calculated in loop.
                        Dimension = n_cal_sources x 3

    cal_pix_inds:       Index (x,y) of pixels closest to cal_sources. Calculated in loop.
                        Dimension = n_cal_sources x 2

    fix_holographic_phase: [Boolean] Temporary fix to undo a phase offset in the holographic images.
                        Especially important when using exclude_autos.
                        Default = True

    flatten_array:      [Boolean] Option to phase model visibilities to zenith to account for non-planarity
                        of array in 2D gridding.
                        Default = False

    conv_thresh:        [float] Threshold for median fractional change in gain in calibration minor loop.
                        Default = 0.01

    conv_max_try:       [integer] Maximum iterations in cal minor loop before quitting due to lack of convergence.
                        Default = 20

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
        n_iter=10, damping_factor=0.0, inv_gains=False, sky_model=NP.ones(1,dtype=NP.float32), cal_sources=None, 
        n_cal_sources=1, phase_fit=False, auto_noise_model=0.0, exclude_autos=False, fix_holographic_phase=True, 
        flatten_array=False, conv_thresh=0.01, conv_max_try=200):

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

        if (damping_factor > 1.0) or (damping_factor < 0.0): damping_factor = 0.0

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
        self.sim_mode = sim_mode
        if sim_mode:
            self.sim_gains = self.simulate_gains()
        self.n_iter = n_iter
        self.count = 0
        self.damping_factor = damping_factor
        self.inv_gains = inv_gains
        self.phase_fit = phase_fit
        self.sky_model = sky_model
        self.auto_noise_model = auto_noise_model
        self.exclude_autos = exclude_autos
        self.auto_corr = NP.zeros((n_ant,n_chan), dtype=NP.float32)
        self.ant_twt = NP.zeros((n_ant,n_chan), dtype=NP.int32)
        if cal_sources is None:
            # Use brightest sources in model
            if self.sky_model.shape[1] > 1:
                arr = sky_model[:,n_chan/2,3]
            else:
                arr = sky_model[:,0,3]
            self.cal_sources = arr.argsort()[-n_cal_sources:][::-1] # returns indices of brightest n_cal_sources sources
            self.n_cal_sources = n_cal_sources
        else:
            cal_sources = NP.array([cal_sources]).flatten()
            self.cal_sources = cal_sources
            self.n_cal_sources = cal_sources.shape[0]

        self.cal_pix_inds = None # placeholder until it can be calculated.

        self.cal_corr = NP.zeros((self.n_cal_sources,n_ant,n_chan), dtype=NP.complex64)
        
        # model_vis, cal_pix_loc, and cal_pix_ind are determined after an imgobj is passed in.
        self.fix_holographic_phase = fix_holographic_phase
        self.flatten_array = flatten_array
        self.conv_thresh = conv_thresh
        self.conv_max_try = conv_max_try

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
        print 'Updating model visibilities.'

        # First find the appropriate pixels to phase to.
        self.cal_pix_inds = NP.zeros((self.n_cal_sources,2))
        self.cal_pix_locs = NP.zeros((self.n_cal_sources,3))
        for i in NP.arange(self.n_cal_sources):
            xind,yind = NP.unravel_index(NP.argmin((gridl-self.sky_model[self.cal_sources[i],0,0])**2+(gridm-self.sky_model[self.cal_sources[i],0,1])**2),gridl.shape) 
            self.cal_pix_inds[i,:] = NP.array([xind,yind])
            self.cal_pix_locs[i,:] = NP.array([gridl[xind,yind], gridm[xind,yind], NP.sqrt(1-gridl[xind,yind]**2-gridm[xind,yind]**2)])

        # Reshape arrays to match: n_src x n_ant x n_chan x 3
        n_src = self.sky_model.shape[0]
        if self.sky_model.shape[1] > 1:
            model_pos = NP.reshape(self.sky_model[:,:,0:3],(n_src,1,self.n_chan,3))
            model_flux = NP.reshape(self.sky_model[:,:,3], (n_src,1,self.n_chan))
        else:
            # Only one channel in model - assume constant flux. Save memory.
            print 'Using same model visibilities for all channels'
            model_pos = NP.reshape(self.sky_model[:,:,0:3],(n_src,1,1,3))
            model_flux = NP.reshape(self.sky_model[:,:,3], (n_src,1,1))

        ant_pos = NP.reshape(self.ant_pos, (1,self.n_ant,self.n_chan,3))

        # Calculate model visibilities
        # TODO: Incorporate beam
        print 'Forming model visibilities'
        model_vis = NP.zeros((self.n_ant, self.n_ant, self.n_chan), dtype=NP.complex128)
        antnum=0
        p = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Antennas '.format(self.n_ant), PGB.ETA()], maxval=self.n_ant).start()
        for ant in xrange(self.n_ant):
            if self.sky_model.shape[1] > 1:
                model_vis[ant,:,:] = NP.sum(model_flux * NP.exp(-1j * 2*NP.pi * NP.sum(model_pos * (ant_pos-NP.reshape(ant_pos[:,ant,:,:],(1,1,self.n_chan,3))),axis=3)),axis=0)
            else:
                model_vis[ant,:,:] = NP.sum(model_flux * NP.exp(-1j * 2*NP.pi * NP.sum(model_pos * (ant_pos[:,:,self.n_chan/2,:].reshape(1,self.n_ant,1,3) - NP.reshape(ant_pos[:,ant,self.n_chan/2,:],(1,1,1,3))),axis=3)),axis=0)
            p.update(antnum+1)
            antnum += 1
        p.finish()

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

        # Add phase to flatten array
        if self.flatten_array:
            for ant in xrange(self.n_ant):
                # TODO: check conjugation
                model_vis[ant,:,:] *= NP.exp(1j * 2*NP.pi * (self.ant_pos[:,:,2]-self.ant_pos[ant,:,2].reshape(1,-1)))

        self.model_vis = model_vis
        print 'Finished updating model visibilities.'

    ######

    def update_cal(self, Edata, imgobj):
        # Check if correlation pixel is known
        if self.cal_pix_inds is None:
            self.update_model_vis(imgobj.gridl, imgobj.gridm)

        #imgdata = imgobj.holimg[self.pol][self.cal_pix_ind[0],self.cal_pix_ind[1],:].flatten()
        imgdata = NP.zeros((self.n_cal_sources,self.n_chan),dtype=NP.complex64)
        for i in xrange(self.n_cal_sources):
            imgdata[i,:] = imgobj.holimg[self.pol][self.cal_pix_inds[i,0],self.cal_pix_inds[i,1],:].flatten()

        self.calc_corr(Edata,imgdata)

        if self.count == self.n_iter:
            # Reached integration level, update the estimated gains

            # get a set of gains for each cal source
            temp_gains = NP.repeat(self.curr_gains.reshape(1,self.n_ant,self.n_chan),self.n_cal_sources,axis=0)
            tries = 0
            change = 100.0 # placeholder

            # TODO:
            # Account for beam value at cal source location!

            # Handle temporary "feature" in the imaging
            if self.fix_holographic_phase:
                # The holographic images have a silly phase running through them due to not centering when padding. Take it out.
                dl = imgobj.gridl[0,1]-imgobj.gridl[0,0]
                dm = imgobj.gridm[1,0]-imgobj.gridm[0,0]
                phase_fix = NP.exp(-1j * NP.pi * (self.cal_pix_locs[:,0]/dl + self.cal_pix_locs[:,1]/dm) / 2).reshape(self.n_cal_sources,1,1)
                self.cal_corr = self.cal_corr * phase_fix            

            self.cal_corr = self.cal_corr / self.ant_twt.reshape((1,self.n_ant,self.n_chan)) # make it an average
            if self.exclude_autos:
                self.auto_corr = self.auto_corr / self.ant_twt

                        # Expression depends on type of calibration
            if self.inv_gains:
                # Inverted gains version
                applied_cal = NP.conj(self.curr_gains).reshape((1,self.n_ant,self.n_chan))
            else:
                # Regular version
                applied_cal = 1/self.curr_gains.reshape((1,self.n_ant,self.n_chan))

            if self.exclude_autos:
                # HACK to add beam
                #self.cal_corr[1,:,:] = self.cal_corr[1,:,:]/NP.sqrt(.57)
                self.cal_corr = self.cal_corr - NP.exp(1j * 2*NP.pi * NP.sum(self.ant_pos[:,:,0:2].reshape((1,self.n_ant,self.n_chan,2)) * self.cal_pix_locs[:,0:2].reshape(self.n_cal_sources,1,1,2),axis=3)) * self.auto_corr.reshape((1,self.n_ant,self.n_chan)) * NP.conj(applied_cal) / self.n_ant
             

            while (tries < self.conv_max_try) and (change > self.conv_thresh):
                # Begin 'minor loop'
                prev_gains = temp_gains
                temp_gains = self.cal_corr * (NP.sum(self.ant_twt,axis=0).reshape(1,1,-1) - self.ant_twt.reshape(1,self.n_ant,self.n_chan)) * self.n_ant / NP.sum((self.n_ant-1) * self.ant_twt.reshape(1,1,self.n_ant,self.n_chan) * NP.exp(1j * 2*NP.pi * NP.sum(self.ant_pos[:,:,0:2].reshape((1,1,self.n_ant,self.n_chan,2)) * self.cal_pix_locs[:,0:2].reshape(self.n_cal_sources,1,1,1,2),axis=4)) * self.model_vis.reshape((1,self.n_ant,self.n_ant,self.n_chan)) * NP.conj(NP.reshape(applied_cal*prev_gains,(self.n_cal_sources,1,self.n_ant,self.n_chan))), axis=2)

                # Average in frequency
                for i in NP.arange(NP.ceil(NP.float(self.n_chan)/self.freq_ave)):
                    mini=i*self.freq_ave
                    maxi=NP.min((self.n_chan,(i+1)*self.freq_ave))
                    temp_gains[:,:,mini:maxi] = NP.nanmean(temp_gains[:,:,mini:maxi],axis=2).reshape(self.n_cal_sources,self.n_ant,1)

                # Fix ref_ant's phase.
                phasor = temp_gains[:,self.ref_ant,:]/NP.abs(temp_gains[:,self.ref_ant,:])
                temp_gains = temp_gains * NP.conj(phasor).reshape(self.n_cal_sources,1,self.n_chan)

                temp_gains = prev_gains * self.damping_factor + temp_gains * (1-self.damping_factor)

                change = NP.median(NP.abs(temp_gains-prev_gains)/NP.abs(prev_gains))
                tries += 1

            print 'Cal minor loop took {} iterations.'.format(tries)
            if tries == self.conv_max_try:
                print 'Warning! Gains failed to converge. Continuing.'

            # Combine gains found from all pixels.
            # For now do the simplest thing and just average. Should probably be a weighted average of sorts eventually.
            temp_gains = NP.mean(temp_gains,axis=0)

            if self.phase_fit:
                # Only fit phase
                temp_gains = temp_gains / NP.abs(curr_gains)

            self.curr_gains = NP.where(NP.isnan(temp_gains),self.curr_gains,self.curr_gains * self.damping_factor + temp_gains * (1 - self.damping_factor))
            if self.phase_fit:
                # phase correction with damping_factor could result in small amplitude drop. Fix this.
                self.curr_gains = self.curr_gains / NP.abs(self.curr_gains)

            # Reset integrations
            self.count = 0
            self.cal_corr = NP.zeros((self.n_cal_sources,self.n_ant,self.n_chan), dtype=NP.complex64)
            self.auto_corr = NP.zeros((self.n_ant,self.n_chan), dtype=NP.float32)
            self.ant_twt = NP.zeros((self.n_ant,self.n_chan), dtype=NP.int32)

    ######

    def calc_corr(self, Edata, imgdata):
        # Perform the correlation of antenna data with image output.
        # Kind of silly to separate this into a function, but conceptually it makes sense.
        #self.cal_corr += NP.where(NP.isnan(Edata),0,Edata*NP.reshape(NP.conj(imgdata),(1,self.n_chan)))
        self.cal_corr += NP.where(NP.isnan(Edata.reshape((1,self.n_ant,self.n_chan))),0,Edata.reshape((1,self.n_ant,self.n_chan))*NP.reshape(NP.conj(imgdata),(self.n_cal_sources,1,self.n_chan)))
        
        self.ant_twt += NP.where(NP.isnan(Edata),0,1)
        if self.exclude_autos:
            #self.auto_corr = self.auto_corr + NP.abs(Edata)**2
            self.auto_corr += NP.where(NP.isnan(Edata),0,NP.abs(Edata)**2)
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


