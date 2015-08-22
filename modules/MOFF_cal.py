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

  sim_gains:      Complex numpy array representing the simulated gains (if sim_mode), 
                  which areapplied to the simulated data.

  cal_method:     String indicating which calibration method/algorithm to use.

  sky_model:      Numpy array representing the sky model to be used for calibration.

  Functions:

  simulate_gains: Create a set of gains for simulation purposes

  update_cal:     Updates the calibration solution given the curr_gains, sky_model, and 
                  input data.

  apply_cal:      Applies current gain solutions to data.

  scramble_gains: Apply a random jitter to the current gains.

  ----------------------------------------------------------------------------------
  """

  def __init__(self, n_ant, n_chan, n_iter=10, cal_method='default', sim_mode=False, sky_model=NP.ones(1,dtype=NP.complex64), ref_ant=0, gain_factor=1.0):

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
      sky_model = NP.repeat(sky_model,n_chan)
    self.sky_model=sky_model

    if cal_method == 'default':
      cal_method = self.default_cal
    self.cal_method = cal_method
    if sim_mode:
      self.sim_mode = True
      self.sim_gains = self.simulate_gains()
    else: self.sim_mode = False

    self.ref_ant = ref_ant


  ####################################

  def simulate_gains(self):

    # For now simple random numbers.
    
    ang=2*NP.pi*NP.random.uniform(low=0.0,high=1,size=[self.n_ant,self.n_chan])
    amp=NP.abs(NP.random.normal(loc=1,scale=0.25,size=[self.n_ant,self.n_chan]))
    
    return amp*NP.exp(1j*ang)

  ##########

  def update_cal(self,*args):
    # For now, assume data should be n_ant x n_chan
    #if Edata.shape != (self.n_ant, self.n_chan):
    #  raise ValueError('Data is the wrong size!')
    # just a one option for now
    self.temp_gains += self.cal_method(*args)
    #self.temp_gains += simple_cal(Edata,self.sky_model,self.curr_gains)
    self.count += 1
    
    if self.count == self.n_iter:
      # reached integration level, update the estimated gains
      self.curr_gains = self.curr_gains*(1-self.gain_factor) + self.gain_factor*self.temp_gains/self.n_iter
      self.count = 0
      self.temp_gains[:] = 0.0

  ###########

  def apply_cal(self,data,meas=False,inv=False):
    if self.curr_gains.size != data.size:
      raise ValueError('Data does not match calibration gain size')

    if meas:
      # in measurement mode, apply sim_gains
      data = data * self.sim_gains
    else:
      # in calibration mode
      if inv:
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
    # Currently it is set up for a calibration where data is multiplied by conj(gains), but this may change in future iterations.
    
    while isinstance(args[0],tuple):
      # we can get here by passing args several times, need to unpack
      args=args[0]
    Edata=args[0]
    imgdata=args[1]*self.n_ant
    if Edata.shape != self.curr_gains.shape:
      raise ValueError('Data does not match calibration gain size')

    #TODO: get this in appropriately
    Ae = 1.0 # effective area

    if self.sky_model.shape[0] != self.n_chan:
      if self.sky_model.shape[0] == 1:
        # just given single model, copy for channels
        model=NP.repeat(self.sky_model,self.n_chan)
      else:
        raise ValueError('Calibration model does not match number of channels')

    new_gains = NP.zeros(self.curr_gains.shape,dtype=NP.complex64)
    for ant in xrange(self.n_ant):
      new_gains[ant,:] = (Edata[ant,:] * NP.conj(imgdata) - self.curr_gains[ant,:]*NP.abs(Edata[ant,:])**2)/(self.sky_model * ( NP.sum(NP.abs(self.curr_gains)**2,axis=0)-NP.abs(self.curr_gains[ant,:])**2))

    return new_gains




