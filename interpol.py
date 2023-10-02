
from scipy import special
import numpy as np
import numpy.fft as t
import matplotlib.pyplot as plt
import scipy.interpolate as interp

import cProfile as profile
import pstats

from mpl_toolkits.axes_grid1 import make_axes_locatable

def interplinear (s, X, Y, geom=0):
	if not np.isscalar(s):
		shape = s.shape
		y_vec = np.arange(shape[0]+1)
		x_vec = np.arange(shape[1]+1)
		s_ag = np.vstack([s, s[0,:]])
		s_ag = np.hstack([s_ag, s_ag[:,0].reshape(shape[0]+1,1)])
		if geom:
			f_s  = interp.RegularGridInterpolator((y_vec, x_vec), 1.0/s_ag)
			return 1.0/f_s(np.array(list(zip(Y, X))))
		else:
			f_s  = interp.RegularGridInterpolator((y_vec, x_vec), s_ag)
			return f_s(np.array(list(zip(Y, X))))
	else:
		return s

def interpspline_entire (s, n, order, correction=None):
	shape = s.shape
	
	if shape[0] == n[0] and  shape[1] == n[1]:
		return s

	y_vec = np.arange(shape[0]+2)
	x_vec = np.arange(shape[1]+2)

	new_y_vec = np.linspace(0.0, shape[0], num=n[0], endpoint=False) + 1
	new_x_vec = np.linspace(0.0, shape[1], num=n[1], endpoint=False) + 1

	if correction != None:
		new_y_vec = new_y_vec + correction[0]
		new_x_vec = new_x_vec + correction[1]

	s_ag = np.vstack([s[-1,:], s, s[0,:]])
	s_ag = np.hstack([s_ag[:,-1].reshape(shape[0]+2,1), s_ag, s_ag[:,0].reshape(shape[0]+2,1)])

	f_s  = interp.RectBivariateSpline(y_vec, x_vec, s_ag, kx=order, ky=order)
	return f_s(new_y_vec, new_x_vec, grid=True).reshape(n)

def interpregular (s, corners, res_coeff, method, correction=None):
	n = (corners[1][0] - corners[0][0], corners[1][1] - corners[0][1])

	corners_orig = [(int(corners[0][0]/res_coeff[0]), int(corners[0][1]/res_coeff[1])), (int(corners[1][0]/res_coeff[0]), int(corners[1][1]/res_coeff[1]))]
	s = s[corners_orig[0][0]-1:corners_orig[1][0]+1, corners_orig[0][1]-1:corners_orig[1][1]+1]
	if res_coeff[0] == 1 and res_coeff[1] == 1:
		return s[1:-1,1:-1]
	else:
		shape = s.shape
		
		y_vec = np.arange(shape[0])
		x_vec = np.arange(shape[1])

		new_y_vec = np.linspace(1.0, shape[0] - 1, num=n[0], endpoint=False)
		new_x_vec = np.linspace(1.0, shape[1] - 1, num=n[1], endpoint=False)

		if correction != None:
			new_y_vec = new_y_vec + correction[0]
			new_x_vec = new_x_vec + correction[1]

		X, Y = np.meshgrid(new_x_vec, new_y_vec)

		# s_ag = np.vstack([s[-1,:], s, s[0,:]])
		# s_ag = np.hstack([s_ag[:,-1].reshape(shape[0]+2,1), s_ag, s_ag[:,0].reshape(shape[0]+2,1)])

		f_s  = interp.RegularGridInterpolator((y_vec, x_vec), s, method=method)
		
		return f_s((Y, X)).reshape(n)

def interpregular_entire (s, n, method, correction=None):
	shape = s.shape
	if shape[0] == n[0] and  shape[1] == n[1]:
		return s
	
	y_vec = np.arange(shape[0]+2)
	x_vec = np.arange(shape[1]+2)

	new_y_vec = np.linspace(0.0, shape[0], num=n[0], endpoint=False) + 1
	new_x_vec = np.linspace(0.0, shape[1], num=n[1], endpoint=False) + 1

	if correction != None:
		new_y_vec = new_y_vec + correction[0]
		new_x_vec = new_x_vec + correction[1]

	X, Y = np.meshgrid(new_x_vec, new_y_vec)

	s_ag = np.vstack([s[-1,:], s, s[0,:]])
	s_ag = np.hstack([s_ag[:,-1].reshape(shape[0]+2,1), s_ag, s_ag[:,0].reshape(shape[0]+2,1)])
	f_s  = interp.RegularGridInterpolator((y_vec, x_vec), s_ag, method=method)
	
	return f_s((Y, X)).reshape(n)

def interpspline (s, corners, res_coeff, order, correction=None):
	n = (corners[1][0] - corners[0][0], corners[1][1] - corners[0][1])

	corners_orig = [(int(corners[0][0]/res_coeff[0]), int(corners[0][1]/res_coeff[1])), (int(corners[1][0]/res_coeff[0]), int(corners[1][1]/res_coeff[1]))]

	s = s[corners_orig[0][0]-1:corners_orig[1][0]+1, corners_orig[0][1]-1:corners_orig[1][1]+1]
	
	if res_coeff[0] == 1 and res_coeff[1] == 1:
		return s[1:-1,1:-1]
	else:
		shape = s.shape

	y_vec = np.arange(shape[0])
	x_vec = np.arange(shape[1])

	new_y_vec = np.linspace(1.0, shape[0] - 1, num=n[0], endpoint=False)
	new_x_vec = np.linspace(1.0, shape[1] - 1, num=n[1], endpoint=False)

	if correction != None:
		new_y_vec = new_y_vec + correction[0]
		new_x_vec = new_x_vec + correction[1]

	# s_ag = np.vstack([s[-1,:], s, s[0,:]])
	# s_ag = np.hstack([s_ag[:,-1].reshape(shape[0]+2,1), s_ag, s_ag[:,0].reshape(shape[0]+2,1)])

	f_s  = interp.RectBivariateSpline(y_vec, x_vec, s, kx=order, ky=order)

	return f_s(new_y_vec, new_x_vec, grid=True).reshape(n)

def interpfftn (s, n, correction=None):
	s_dims = s.shape
	if s_dims[0] == n[0] and  s_dims[1] == n[1]:
		return s

	x_step = int((s_dims[1]-1)/n[1]) + 1
	y_step = int((s_dims[0]-1)/n[0]) + 1

	new_n = [0 , 0]
	new_n[1] = n[1] * x_step
	new_n[0] = n[0] * y_step

	fact = (new_n[0]*new_n[1])/(s_dims[0]*s_dims[1])
	
	new_center = [0 ,0]
	new_center[1] = int(np.floor(new_n[1]/2))
	new_center[0] = int(np.floor(new_n[0]/2))
	
	if s_dims == new_n:
		s_padded = s
	else:
		fft_s = t.fftn(s)
		if correction is not None:
			fft_s = t.fftshift(correction * fft_s)
		else:
			fft_s = t.fftshift(fft_s)

		fft_s_padded = np.zeros([new_n[0],new_n[1]] , dtype=complex)

		fft_s_padded[new_center[0] - int(s_dims[0]/2): new_center[0] + int(np.ceil(s_dims[0]/2)), 
					 new_center[1] - int(s_dims[1]/2): new_center[1] + int(np.ceil(s_dims[1]/2))] = fft_s*fact
		
		s_padded = np.ascontiguousarray(np.real(t.ifftn(t.ifftshift(fft_s_padded))))

	ret =  np.copy(s_padded[0:new_n[0]:y_step, 0:new_n[1]:x_step])
	return ret

def interpfftn_print (s, n, correction=None):
	s_dims = s.shape
	if s_dims == n:
		return s

	x_step = int((s_dims[1]-1)/n[1]) + 1
	y_step = int((s_dims[0]-1)/n[0]) + 1

	new_n = [0 , 0]
	new_n[1] = n[1] * x_step
	new_n[0] = n[0] * y_step

	fact = (new_n[0]*new_n[1])/(s_dims[0]*s_dims[1])
	
	new_center = [0 ,0]
	new_center[1] = int(np.floor(new_n[1]/2))
	new_center[0] = int(np.floor(new_n[0]/2))
	
	if s_dims == new_n:
		s_padded = s
	else:
		fft_s = t.fftn(s)
		if correction is not None:
			fft_s = t.fftshift(correction * fft_s)
		else:
			fft_s = t.fftshift(fft_s)

		fft_s_padded = np.zeros([new_n[0],new_n[1]] , dtype=complex)

		fft_s_padded[new_center[0] - int(s_dims[0]/2): new_center[0] + int(np.ceil(s_dims[0]/2)), 
					 new_center[1] - int(s_dims[1]/2): new_center[1] + int(np.ceil(s_dims[1]/2))] = fft_s*fact
		
		s_padded = np.ascontiguousarray(np.real(t.ifftn(t.ifftshift(fft_s_padded))))

	ret =  np.copy(s_padded[0:new_n[0]:y_step, 0:new_n[1]:x_step])

	# fig  = plt.figure(14)
	# ax1  = fig.add_subplot(121)
	# ax2  = fig.add_subplot(122)
	# ax1.imshow(s)
	# ax2.imshow(ret)
	# plt.show()

	return ret

def main():
	# x = np.array([1, 2, 3, 5, 3, 2, 1, 2])
	# cos = np.cos(np.arange(0,31)*2*np.pi/31)
	# signal = np.tile(cos,(31,1)) * np.tile(cos.reshape(31,1),(1,31))
	# sig_min = np.min(signal)
	# sig_max = np.max(signal)
	# plt.figure()
	# plt.imshow(signal,                   vmin=sig_min, vmax=sig_max)
	# plt.show()
	# exit(0)

	N = 64
	signal = np.random.rand(N,N)*10

	# prof = profile.Profile()
	# prof.enable()
	res1 = interpregular_entire(signal, (128,128), "slinear")
	res1 = res1[12:24,12:24]

	res2 = interpregular       (signal, [(12,12),(24,24)], [2,2], "slinear")

	print(res1.dtype)
	print(res2.dtype)
	# prof.disable()

	# stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
	# stats.print_stats(1)
	# stats.print_callees(1)

	fig = plt.figure()

	ax1  = fig.add_subplot(221)
	ax2  = fig.add_subplot(222)
	ax3  = fig.add_subplot(223)

	sig_min = np.min(signal)
	sig_max = np.max(signal)
	
	
	img1 = ax1.imshow(res1, vmin=sig_min, vmax=sig_max)
	
	divider = make_axes_locatable(ax1)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(img1, cax=cax, orientation='vertical')
	
	img2 = ax2.imshow(res2, vmin=sig_min, vmax=sig_max)
	
	divider = make_axes_locatable(ax2)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(img2, cax=cax, orientation='vertical')

	img3 = ax3.imshow((res2 - res1))
	
	divider = make_axes_locatable(ax3)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(img3, cax=cax, orientation='vertical')

	plt.show()

if __name__ == '__main__':
	main()

