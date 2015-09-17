import numpy as NP
import ipdb as PDB
import scipy.constants as FCNST


class cal:
    
    """
    -------------------------------------------------------------------------------
    Class to handle everything calibration for MOFF
  
    *** Attributes ***
  
    n_ant:          Number of antennas. Calculated from ant_pos.
    
    n_chan:         Number of frequency channels. Calculated from freqs.
    
    curr_gains:     Complex numpy array representing current estimate of gains.
                    Dimensions n_ant x n_chan
    
    n_iter:         [integer] Number of iterations before updating the calibration.
    
    count:          [integer] Current iteration number.
    
    gain_factor:    Weighting factor when updating gains.
    
    temp_gains:     Complex numpy array, gains currently being integrated/averaged.
    
    sim_mode:       [Boolean] representing whether in simulation mode. Default = False.
    
    sim_gains:      Complex numpy array representing the simulated gains (if sim_mode), 
                    which are applied to the simulated data.
    
    cal_method:     String indicating which calibration method/algorithm to use.
                    Current options: 'single' (single source), 'multi' (multi source).
    
    inv_gains:      [Boolean] If False (default), calibrate by dividing by gains. If 
                    True, calibrate by multiplying by conjugate of gains.
    
    sky_model:      Numpy array containing l,m,n,flux for sources in sky model for each 
                    channel. Dimensions should be n_source x n_chan x 4.

    model_vis:      Model visibilities if cal_method='multi'. Calculated in init.
                    Dimensions are n_ant x n_ant x n_chan.

    ant_pos:        Numpy array with antenna locations in wavelengths for each frequency. 
                    Dimesions should be n_ant x n_chan x 3. If n_ant x 3, assumed to be 
                    in meters and rebroadcasted in init.

    ref_ant:        Antenna index to fix phase.

    ref_point:      Numpy array of reference sky point for phasing of antennas. 
                    Default is (l,m,n) = (0,0,1)

    freqs:          Numpy array with frequencies in Hz.

    pol:            Polarization key (to reference image object)

    cal_ver:        Calibration version. To keep track of different cal loops.
    
    *** Functions ***
    
    simulate_gains: Create a set of gains for simulation purposes
    
    update_cal:     Updates the calibration solution given the curr_gains, sky_model, and 
                    input data.
    
    apply_cal:      Applies current gain solutions to data.
    
    scramble_gains: Apply a random jitter to the current gains. Useful for testing/simulating.

    simple_cal:     Calibrates for a single point source on the center of the field.

    off_center_cal: Calibrates for single point source in arbitrary location.
    
    ----------------------------------------------------------------------------------
    """

    def __init__(self, ant_pos, freqs, n_iter=10, cal_method='default', sim_mode=False, sky_model=NP.ones(1,dtype=NP.complex64), ref_ant=0, gain_factor=1.0, inv_gains=False, pol='P1',ref_point=NP.array([0,0,1]),cal_ver='new'):

        # Get derived values and check types, etc.
        n_chan=freqs.shape[0]
        n_ant=ant_pos.shape[0]
        if not ant_pos.shape[-1] == 3:
            raise ValueError('Antenna positions much be three dimensional!')
        if len(ant_pos.shape) != 3:
            if len(ant_pos.shape) == 2:
                print 'Antennas positions assumed to be in meters. Reformatting to wavelengths'
                ant_pos = ant_pos.reshape(n_ant,1,3)
                ant_pos = ant_pos * freqs.reshape(1,n_chan,1)/FCNST.c
            else:
                ValueError('Antenna positions in wrong format!')
        elif ant_pos.shape[1] != n_chan:
            ValueError('Antenna positions do not match frequencies!')
        
        if not isinstance(n_iter,int):
            raise TypeError('n_iter must be an integer')
        elif n_iter <=0:
            n_iter=1
        if not isinstance(cal_method,str):
            print 'cal_method must be a string. Using default'
            cal_method='default'

        if (gain_factor > 1.0) or (gain_factor < 0.0): gain_factor = 1.0
        if len(ref_point.shape) == 2:
            # Assume only (l,m) are given.
            ref_point=NP.hstack((ref_point,NP.sqrt(1-NP.sum(ref_point**2))))

        switcher = {
            'default': self.default_cal,
            'simple': self.simple_cal,
            'off_center': self.off_center_cal,
            'multi_source': self.multi_source_cal,
        }

        # Defaults for sky model:
        #   If no spectral info given, assume constant
        #   If no position given, assume on center
        if len(sky_model.shape) == 1:
            if sky_model.shape[0] == 1:
                # Assume just given flux.
                temp_model = NP.zeros((1,self.n_chan,4))
                temp_model[0,:,3] = sky_model
                temp_model[0,:,2] = 1
                sky_model = temp_model
            elif sky_model.shape[0] == 4:
                # Assume given postion,flux correctly.
                sky_model = NP.tile(sky_model,(1,n_chan,1))
            elif sky_model.shape[0] == 3:
                # Assume position,flux is given by (l,m,flux) (missing n)
                temp_model = NP.zeros(4)
                temp_model[0:2] = sky_model[0,:]
                temp_model[2] = NP.sqrt(1-NP.sum(temp_model[0:2]**2))
                temp_model[3] = sky_model[2]
                sky_model = NP.tile(temp_model,(1,n_chan,1))
            else:
                ValueError('Unrecognized sky model format!')
        elif len(sky_model.shape) != 3:
            # TODO: more cases. But this should be ok for now
            ValueError('Unrecognized sky model format!')
        
                
        # Assign to attributes    
        self.n_ant = n_ant
        self.n_chan = n_chan
        self.curr_gains = NP.ones([n_ant,n_chan], dtype=NP.complex64)
        self.n_iter = n_iter
        self.inv_gains = inv_gains
        self.gain_factor = gain_factor
        self.ref_ant = ref_ant
        self.ref_point = ref_point
        self.pol = pol
        self.cal_ver = cal_ver
        
        self.sky_model=sky_model
        self.ant_pos=ant_pos
        self.cal_method = switcher.get(cal_method,self.default_cal)
        if sim_mode:
            self.sim_mode = True
            # initialize simulated gains
            self.sim_gains = self.simulate_gains()
        else: self.sim_mode = False

        self.temp_gains = NP.zeros([n_ant,n_chan], dtype=NP.complex64)
        self.count = 0

        if self.cal_method == self.multi_source_cal:
            self.gen_model_vis()



    ####################################

    def simulate_gains(self,scale=0.25):

        # For now simple random numbers.
    
        ang=2*NP.pi*NP.random.uniform(low=0.0,high=1,size=[self.n_ant,self.n_chan])
        amp=NP.abs(NP.random.normal(loc=1,scale=scale,size=[self.n_ant,self.n_chan]))
    
        return amp*NP.exp(1j*ang)

    ##########

    def update_cal(self,*args):
        # just one option for now
        self.temp_gains += self.cal_method(*args)
        self.count += 1
        
        if self.count == self.n_iter:
            # reached integration level, update the estimated gains
            self.temp_gains /= self.n_iter
            self.curr_gains = self.curr_gains*(1-self.gain_factor) + self.gain_factor*self.temp_gains
            self.count = 0
            self.temp_gains[:] = 0.0

    ###########

    def apply_cal(self,data,meas=False):
        if self.curr_gains.size != data.size:
            raise ValueError('Data does not match calibration gain size')

        if meas:
            # in measurement mode, apply sim_gains
            data = data * self.sim_gains
        else:
            # in calibration mode
            if self.inv_gains:
                # apply inverse gains - i.e. multiply by gain to calibrate
                data = data * NP.conj(self.curr_gains)
            else:
                data = data / self.curr_gains

        return data
    
    ######
    
    def scramble_gains(self,amp):
        # mix up the gains a bit
        self.curr_gains += NP.random.normal(0,NP.sqrt(amp),self.curr_gains.shape) + 1j * NP.random.normal(0,NP.sqrt(amp),self.curr_gains.shape)
        
        return

    #######

    def gen_model_vis(self):
        # Function to generate the model visibilites, given sky model and ant positions
        # This will eventually move out to Nithya's modules to account for beams,
        # but for testing we'll just do it here.

        # first reshape arrays to match: n_src x n_ant x n_chan x 3
        n_src = self.sky_model.shape[0]
        model_pos = NP.reshape(self.sky_model[:,:,0:3]-self.ref_point.reshape((1,1,3)),(n_src,1,self.n_chan,3))
        model_flux = NP.reshape(self.sky_model[:,:,3],(n_src,1,self.n_chan))
        ant_pos = NP.reshape(self.ant_pos,(1,self.n_ant,self.n_chan,3))
        
        model_vis = NP.zeros((self.n_ant,self.n_ant,self.n_chan),dtype=NP.complex128)
        for ant in xrange(self.n_ant):
            # TODO: check conjugation
            model_vis[ant,:,:] = NP.sum(model_flux*NP.exp(-1j * 2*NP.pi * NP.sum(model_pos*(ant_pos-NP.reshape(ant_pos[:,ant,:,:],(1,1,self.n_chan,3))),axis=3)),axis=0)

        self.model_vis = model_vis
        
    """
    ------------
    Calibration method functions
    ------------
    """

    def default_cal(self,*args):
        # point to whatever routine is currently the default
        out = self.off_center_cal(args)
        return out
  
    ######

    def simple_cal(self,*args):
        #
        ### Simple Calibration
        # Assumes a single point source on the center pixel.
        # Can be run when data is multiplied by conj gains, or when divided by gains.
        #
        # Inputs:
        #   Edata:    _uncalibrated E-field data from antennas.
        #   imgdata:  Instantaneous pixel (all channels) from holographic image.
        #   model:    Model flux for source (all channels).
        
        while isinstance(args[0],tuple):
            # we can get here by passing args several times, need to unpack
            args=args[0]
        Edata=args[0]
        imgdata=args[1]*self.n_ant
        model=args[2]
        if Edata.shape != self.curr_gains.shape:
            raise ValueError('Data does not match calibration gain size')

        #TODO: get this in appropriately
        Ae = 1.0 # effective area

        if model.shape[0] != self.n_chan:
            raise ValueError('Calibration model does not match number of channels')

        new_gains = NP.zeros(self.curr_gains.shape,dtype=NP.complex64)
        for ant in xrange(self.n_ant):
            if self.cal_ver=='new':
                if self.inv_gains:
                    new_gains[ant,:] = Edata[ant,:]*NP.conj(imgdata) / (model * NP.sum(NP.abs(self.curr_gains)**2))
                else:
                    new_gains[ant,:] = Edata[ant,:]*NP.conj(imgdata) / (model * self.n_ant)
            else:
                if self.inv_gains:
                    new_gains[ant,:] = (Edata[ant,:] * NP.conj(imgdata) - self.curr_gains[ant,:]*NP.abs(Edata[ant,:])**2)/(model * ( NP.sum(NP.abs(self.curr_gains)**2,axis=0)-NP.abs(self.curr_gains[ant,:])**2))
                else:
                    new_gains[ant,:] = 1/(model * (imgdata - Edata[ant,:]/self.curr_gains[ant,:]) / (Edata[ant,:] * (NP.sum(NP.abs(Edata/self.curr_gains)**2,axis=0) - NP.abs(Edata[ant,:]/self.curr_gains[ant,:])**2)))
                    # The version below is incorrect.
                    # But it gets pretty darn close, without any model of the sky.
                    # So I'm going to leave it for now to study later.
                    #new_gains[ant,:] = 1/((imgdata - Edata[ant,:]/self.curr_gains[ant,:]) / (Edata[ant,:] * (NP.sum(NP.abs(1/self.curr_gains)**2,axis=0) - NP.abs(1/self.curr_gains[ant,:])**2)))
                
        phasor=new_gains[self.ref_ant,:]/NP.abs(new_gains[self.ref_ant,:])
        new_gains *= NP.conj(phasor).reshape(1,self.n_chan)
        return new_gains

    ######

    def off_center_cal(self,*args):
        #
        ### Calibrate on single source off-center.
        # Inputs
        #   Edata:    _uncalibrated E-field data from antennas.
        #   imgobj:   Image object output from latest correlator image.
        #   sourcei:  Source number to pull out of self.sky_model (zero indexed)

        while isinstance(args[0],tuple):
            args=args[0]
        Edata = args[0]
        imgobj = args[1]
        sourcei = args[2]

        # rephase the antenna data
        model_pos = self.sky_model[sourcei,:,0:3]-self.ref_point.reshape((1,1,3))
        model_pos = model_pos.reshape(1,self.n_chan,3) # 1 x n_chan x 3
        model_flux = self.sky_model[sourcei,:,3].flatten()
        Edata *= NP.exp(-1j * 2*NP.pi * NP.sum(model_pos*self.ant_pos,axis=2))
        
        # Now get the pixel of interest
        xind,yind = NP.unravel_index(NP.argmin((imgobj.gridl-model_pos[0,0,0])**2 + (imgobj.gridm-model_pos[0,0,1])**2),imgobj.gridl.shape)
        imgdata = imgobj.holimg[self.pol][xind,yind,:].flatten()
        new_gains = self.simple_cal(Edata,imgdata,model_flux)

        return new_gains

    # Thid multi-source cal is now obsolete (and never really worked).    
    """
    def multi_source_cal(self,*args):
        #
        ### Calibrate on several sources simultaneously.
        # Inputs
        #   Edata:    _uncalibrated E-field data from antennas.
        #   imgobj:   Image object output from latest correlator image.

        while isinstance(args[0],tuple):
            args=args[0]
        Edata = args[0]
        imgobj = args[1]

        n_src = self.sky_model.shape[0]
        new_gains = NP.zeros((self.n_ant,self.n_chan),dtype=NP.complex64)
        
        weights = NP.reshape(self.sky_model[:,:,3],(n_src,self.n_chan))
        
        for sourcei in xrange(n_src):
            new_gains += self.off_center_cal(Edata,imgobj,sourcei)*NP.reshape(self.sky_model[sourcei,:,3],(1,self.n_chan))

        
        new_gains /= NP.reshape(NP.sum(weights,axis=0),(1,self.n_chan))

        return new_gains
    """

    def multi_source_cal(self,*args):
        #
        ### Calibrate on several sources simultaneously.
        # Inputs
        #   Edata:    _uncalibrated E-field data from antennas.
        #   imgobj:   Image object output from latest correlator image.

        while isinstance(args[0],tuple):
            args=args[0]
        Edata = args[0]
        imgobj = args[1]

        xind,yind = NP.unravel_index(NP.argmin((imgobj.gridl)**2+(imgobj.gridm)**2),imgobj.gridl.shape) # will eventually generalize this to use any pixel of interest
        imgdata = imgobj.holimg[self.pol][xind,yind,:].flatten()
        
        new_gains = NP.zeros((self.n_ant,self.n_chan),dtype=NP.complex64)

        # TODO:
        # push the sum of vis and gains outside of the integration
        # intelligently select pixel to correlate
        if self.inv_gains:
            # inverted gains version
            new_gains = self.n_ant * Edata * NP.reshape(NP.conj(imgdata),(1,self.n_chan)) / NP.sum(self.model_vis * NP.reshape(NP.abs(self.curr_gains)**2,(1,self.n_ant,self.n_chan)),axis=1)
        else:
            # regular version
            new_gains = self.n_ant * Edata * NP.reshape(NP.conj(imgdata),(1,self.n_chan)) / NP.sum(self.model_vis,axis=1)

        return new_gains
            
        




