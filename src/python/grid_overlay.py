# from lsl.common.stations import lwasv
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl
from astropy.wcs import WCS
from astropy.io import fits
# from astropy.visualization.wcsaxes import WCSAxes
# from astropy.coordinates import Angle
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, FK5
from astropy.time import Time, TimeDelta
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.coordinates import get_body

import click


lwasv_lat = 0.5994919397537344
lwasv_lon = -1.8655088369221824

@click.command()
@click.option('--img-size', default=128, help='Size of the image in pixels.')
@click.option('--file-name', default="epic_grid_overlay.png", help='Name of the output file.')
@click.option('--video-size', default=512, help='Size of the video in pixels.')
@click.option('--resolution', default=1.056, help='Resolution of EPIC images.')
@click.option('--dpi', default=100, help='DPI (dots per inch) for the output image.')
def create_grid_overlay(img_size,file_name,video_size, resolution,dpi):
  # create an image header to build wcs object
  img_data = np.zeros((img_size,img_size))
  ihdu = fits.ImageHDU(img_data)

  # get the ra dec for the zenith
  lwasv_location = EarthLocation(lat=lwasv_lat*180/np.pi*u.deg, lon=lwasv_lon*180/np.pi*u.deg, height=0)
  t0 = Time(Time.now(),location=lwasv_location)
  lsts = t0.sidereal_time("apparent")

  zenith_radec  = SkyCoord(
        lsts.deg, lwasv_lat * 180.0 / np.pi, obstime=t0, unit="deg"
    ).transform_to(FK5(equinox=Time("J2000")))

  ihdu.header["DATETIME"] = t0.isot
  ihdu.header["LST"] = lsts.hour
  ihdu.header["CTYPE1"] = "RA---SIN"
  ihdu.header["CRPIX1"] = img_size/2 + 1
  ihdu.header["CDELT1"] = -resolution
  ihdu.header["CRVAL1"] = zenith_radec.ra.deg
  ihdu.header["CUNIT1"] = "deg"

  ihdu.header["CTYPE2"] = "DEC--SIN"
  ihdu.header["CRPIX2"] = img_size/2 +1 - (img_size/2 + 1)%2
  ihdu.header["CDELT2"] = resolution
  ihdu.header["CRVAL2"] = zenith_radec.dec.deg
  ihdu.header["CUNIT2"] = "deg"

  ihdu.header["EQUINOX"] = 2000
  ihdu.header["RADESYS"] = "FK5"

  wcs = WCS(ihdu.header)
  center=wcs.pixel_to_world(img_size/2+0.5,img_size/2+0.5)
  fig = plt.figure(figsize=(video_size/dpi,video_size/dpi),dpi=dpi)
  ax = fig.add_subplot(111, projection=wcs,frame_on=False)
  ax.coords.frame.set_linewidth(0)

  ax.imshow(img_data, cmap='gray', origin='lower',alpha=0)
  overlay = ax.get_coords_overlay('fk5')
  lon = overlay['ra']
  lat = overlay['dec']

  lon.add_tickable_gridline('const-ra', center.ra-20*u.deg)
  lat.add_tickable_gridline('const-dec', center.dec-20*u.deg)

  lon.set_ticks_visible(False)
  lat.set_ticks_visible(False)

  lon.set_axislabel_visibility_rule('ticks')
  lat.set_axislabel_visibility_rule('ticks')

  lon.set_ticks_position(('const-dec', 'l'))
  lon.set_ticks(spacing=60*u.deg)
  lon.set_ticklabel(size=10,color='white',pad=2,exclude_overlapping=True,rotation=75,rotation_mode='anchor',alpha=0.5)
  lon.set_ticklabel_position(('const-dec',))
  #lon.set_axislabel_position('b')
  #lon.set_format_unit('deg',decimal=True)
  lon.set_major_formatter('hh')
  #lon.get_tick_labels()
  #lon.set_ticklabel(color='red', size=6)
  #lon.set_axislabel_position('r')
  #lon.set_axislabel('Right Ascension', color='red')

  #lat.set_ticks_position(('const-ra', 'r'))
  #lat.set_format_unit('dd',decimal=True)
  lat.set_major_formatter('dd')
  lat.set_ticks_position(('const-ra', ))
  lat.set_ticks(spacing=30*u.deg)
  lat.set_ticklabel(size=10,color='white',pad=4,exclude_overlapping=True,alpha=0.5)
  #lat.set_ticks(color='magenta')
  lat.set_ticklabel_position(('const-ra',))
  

  # reduce the radius slightly to ensure it won't go out of bounds 
  # due to numerical errors
  r = SphericalCircle(center, 90*0.97 * u.degree, 
                      edgecolor='white', facecolor='none',
                      transform=ax.get_transform('fk5'),alpha=0.7)
  ax.add_patch(r)
  galactic_longitudes = np.linspace(0, 360, 1000)
  galactic_latitudes = np.zeros_like(galactic_longitudes)-10*np.pi/180

  # Convert galactic coordinates to sky coordinates (RA, Dec)
  coords_mw = SkyCoord(l=galactic_longitudes*u.degree, b=galactic_latitudes*u.degree, frame='galactic')
  sky_coords_mw = coords_mw.transform_to('fk5')  # Convert to RA/Dec (ICRS frame)

  text_kws={"horizontalalignment":"right","verticalalignment":"bottom","fontweight":"demibold","color":"white"}
# Plot the galactic plane curve
  ax.plot(sky_coords_mw.ra.deg, sky_coords_mw.dec.deg, transform=ax.get_transform('fk5'), color='red', lw=1, ls='--', label="Milky Way",alpha=0.8)

  # Cyg A
  ax.text_coord(SkyCoord(ra=299.8681523682083*u.deg,dec=40.73391589791667*u.deg,frame='fk5'),s="Cyg A",**text_kws)
  # Cas A
  ax.text_coord(SkyCoord(ra=350.85*u.deg,dec=58.815*u.deg,frame='fk5'),s="Cas A",**text_kws)
  # Sun
  sun_pos = get_body('sun',t0)
  ax.text_coord(SkyCoord(ra=sun_pos.ra,dec=sun_pos.dec,frame='fk5'),s='Sun',**text_kws)
  # Jupiter
  jup_pos = get_body('jupiter',t0)
  ax.text_coord(SkyCoord(ra=jup_pos.ra,dec=jup_pos.dec,frame='fk5'),s='Jupiter',**text_kws)
  # M87
  ax.text_coord(SkyCoord(ra=187.705931*u.deg,dec=12.391123*u.deg,frame='fk5'),s="M87",ma='center',**text_kws)
  # Crab
  ax.text_coord(SkyCoord(ra=83.633107*u.deg,dec=22.014486*u.deg,frame='fk5'),s="Tau A",**text_kws)

  overlay.grid(color='white', ls='solid', alpha=0.5) 


  fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0) 
  fig.savefig(file_name, transparent=True,pad_inches=0,dpi=dpi)

if __name__ == '__main__':
    create_grid_overlay()