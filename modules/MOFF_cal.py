import numpy as NP
import ipdb as PDB


class cal:
    
    """
    -------------------------------------------------------------------------------
    Class to handle everything calibration for MOFF
  
    Attributes:
  
    n_ant:          Number of antennas.
    
    n_chan:         Number of frequency channels.
    
    curr_gains:     Complex numpy array representing current estimate of gains.
    
    n_iter:         [integer] Number of iterations before updating the calibration
    
    count:          [integer] Current iteration number
    
    gain_factor:    Weighting factor when updating gains
    
    temp_gains:     Complex numpy array, gains currently being integrated/averaged
    
    sim_mode:       [Boolean] representing whether in simulation mode. Default = False.
    
    sim_gains:      Complex numpy array representing the simulated gains (if sim_mode), which areapplied to the simulated data.
    
    cal_method:     String indicating which calibration method/algorithm to use.
    
    inv_gains:      [Boolean] If False (default), calibrate by dividing by gains. If True, calibrate by multiplying by conjugate of gains.
    
    sky_model:      Numpy array containing l,m,flux for sources in sky model for each channel. Dimensions should be 3 x n_source x n_chan.
    
    Functions:
    
    simulate_gains: Create a set of gains for simulation purposes
    
    update_cal:     Updates the calibration solution given the curr_gains, sky_model, and input data.
    
    apply_cal:      Applies current gain solutions to data.
    
    scramble_gains: Apply a random jitter to the current gains.
    
    ----------------------------------------------------------------------------------
    """

    def __init__(self, n_ant, n_chan, n_iter=10, cal_method='default', sim_mode=False, sky_model=NP.ones(1,dtype=NP.complex64), ref_ant=0, gain_factor=1.0, inv_gains=False):
    
        if not isinstance(n_ant,int):
            raise TypeError('n_ant must be an integer')
        if not isinstance(n_chan,int):
            raise TypeError('n_chan must be an integer')
        elif n_chan <= 0:
            n_chan=1
            
        if not isinstance(n_iter,int):
            raise TypeError('n_iter must be an integer')
        elif n_iter <=0:
            n_iter=1
        if not isinstance(cal_method,str):
            print 'cal_method must be a string. Using default'
            cal_method='default'
    
        self.n_ant = n_ant
        self.n_chan = n_chan
        self.curr_gains = NP.ones([n_ant,n_chan], dtype=NP.complex64)
        self.n_iter = n_iter
        self.count = 0
        if (gain_factor > 1.0) or (gain_factor < 0.0): gain_factor = 1.0
        self.gain_factor = gain_factor
        self.temp_gains = NP.zeros([n_ant,n_chan], dtype=NP.complex64)
        if sky_model.shape[0] == 1:
            # Build up the default sky model
            temp_model = NP.zeros((3,1,self.n_chan))
            temp_model[2,0,:] = sky_model
            sky_model = temp_model
        self.sky_model=sky_model
        self.inv_gains = inv_gains

        switcher = {
            'default': self.default_cal,
            'simple': self.simple_cal,
            'off_center': self.off_center_cal,
        }
        self.cal_method = switcher.get(cal_method,self.default_cal)
        
        if sim_mode:
            self.sim_mode = True
            self.sim_gains = self.simulate_gains()
        else: self.sim_mode = False

        self.ref_ant = ref_ant


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
            if not self.inv_gains:
                # Note that temp gains are actually 1/gains in this mode
                self.temp_gains = 1/self.temp_gains
            self.curr_gains = self.curr_gains*(1-self.gain_factor) + self.gain_factor*self.temp_gains/self.n_iter
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
    """
    ------------
    Calibration method functions
    ------------
    """

    def default_cal(self,*args):
        # point to whatever routine is currently the default
        out = self.simple_cal(args)
        return out
  
    ######

    def simple_cal(self,*args):
        ### Simple Calibration
        # Assumes a single point source on the center pixel.
        # Flux of source should be given in self.sky_model
        # Needs the _uncalibrated_ E-field data from the antennas, and the instantaneous center pixel from the holographic image.
        # Can be run when data is multiplied by conj gains, or when divided by gains.
        while isinstance(args[0],tuple):
            # we can get here by passing args several times, need to unpack
            args=args[0]
        Edata=args[0]
        imgdata=args[1]*self.n_ant
        model=args[2] # Now passed in with single source
        if Edata.shape != self.curr_gains.shape:
            raise ValueError('Data does not match calibration gain size')

        #TODO: get this in appropriately
        Ae = 1.0 # effective area

        if model.shape[0] != self.n_chan:
            raise ValueError('Calibration model does not match number of channels')

        new_gains = NP.zeros(self.curr_gains.shape,dtype=NP.complex64)
        for ant in xrange(self.n_ant):
            if self.inv_gains:
                new_gains[ant,:] = (Edata[ant,:] * NP.conj(imgdata) - self.curr_gains[ant,:]*NP.abs(Edata[ant,:])**2)/(model * ( NP.sum(NP.abs(self.curr_gains)**2,axis=0)-NP.abs(self.curr_gains[ant,:])**2))
            else:
                new_gains[ant,:] = model * (imgdata - Edata[ant,:]/self.curr_gains[ant,:]) / (Edata[ant,:] * (NP.sum(NP.abs(Edata/self.curr_gains)**2,axis=0) - NP.abs(Edata[ant,:]/self.curr_gains[ant,:])**2))
                # The version below is incorrect.
                # But it gets pretty darn close, without any model of the sky.
                # So I'm going to leave it for now to study later.
                #new_gains[ant,:] = (imgdata - Edata[ant,:]/self.curr_gains[ant,:]) / (Edata[ant,:] * (NP.sum(NP.abs(1/self.curr_gains)**2,axis=0) - NP.abs(1/self.curr_gains[ant,:])**2))
        
        return new_gains

    ######

    def off_center_cal(self,*args):
        ### Another simple calibration, but for an off-center calibrator.
        # args should be in the following format:
        # Edata (uncalibrated E-field data), imgdata (pixel closest to calibrator),
        #   model (position and flux for single source), ant_info (antenna
        #   positions in wavelengths)

        while isinstance(args[0],tuple):
            args=args[0]
        Edata = args[0]
        imgdata = args[1]
        model = args[2] # expected to have dimensions 3 x 1 x n_chan
        ant_info = args[3]

        # rephase the antenna data
        model_pos = NP.squeeze(model[0:2,0,:])
        model_pos = NP.tile(model_pos,[self.n_ant,1,1])
        model_flux = model[2,0,:]
        #TODO: Add frequency dependence!
        ant_info = NP.swapaxes(NP.swapaxes(NP.tile(ant_info,[self.n_chan,1,1]),0,1),1,2)
        Edata *= NP.exp(-1j * 2*NP.pi * (model_pos[:,0,:]*ant_info[:,0,:]+model_pos[:,1,:]*ant_info[:,1,:]))

        new_gains = self.simple_cal(Edata,imgdata,model_flux)

        return new_gains




