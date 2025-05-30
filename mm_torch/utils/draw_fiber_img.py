	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
	import numpy as np
	import scipy.ndimage as ndimage
	from skimage.restoration import unwrap_phase

	def plot_fiber(raw_azimuth, linr=10, intensity=None, mask=None, window=5, n=10, option='quiver'):

		azimuth = np.pi*raw_azimuth/180
		X, Y = np.meshgrid(np.arange(azimuth.shape[1]), np.arange(azimuth.shape[0]))

		azimuth_unwrapped = unwrap_phase(azimuth-np.pi)
		orientation_cos = np.cos(azimuth_unwrapped)
		orientation_sin = np.sin(azimuth_unwrapped)

		cos_mean = ndimage.uniform_filter(orientation_cos, (window, window))/window**2
		sin_mean = ndimage.uniform_filter(orientation_sin, (window, window))/window**2
		magnitude = (cos_mean**2 + sin_mean**2)**.5

		if mask is not None: raw_azimuth = np.ma.masked_where(mask, raw_azimuth)
		if mask is not None: orientation_cos = np.ma.masked_where(mask, orientation_cos)
		if mask is not None: orientation_sin = np.ma.masked_where(mask, orientation_sin)
		u = magnitude * orientation_cos * linr
		v = magnitude * orientation_sin * linr

		fig, ax = plt.subplots()
		if intensity is not None:
			intensity = (intensity-intensity.min())/(intensity.max()-intensity.min())
			ax.imshow(intensity, cmap = 'gray')

		if option == 'quiver':
			tracts = ax.quiver(
				X[::n, ::n], Y[::n, ::n], u[::n, ::n], v[::n, ::n],
				raw_azimuth[::n, ::n],
				scale = 20, 
				headwidth = 2, 
				cmap = plt.cm.twilight_shifted,
			)

		if option == 'streamplot':
			u = +cos_mean_masked
			v = -sin_mean_masked
			tracts = ax.streamplot(
				X, Y, u, v, color=azimuth_unwrapped, 
				density = 2, 
				cmap = plt.cm.twilight_shifted, 
				#norm = normalizer, 
				arrowsize = 0, 
				linewidth = 0.1*linr, 
				maxlength = 20., 
			)

		ax.set_axis_off()
		ax.axis('off')
		plt.tight_layout()
		canvas = FigureCanvas(fig)
		canvas.draw()

		# numpy array conversion
		w, h = fig.get_size_inches() * fig.get_dpi()
		quiver_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(h), int(w), 3)

		plt.close(fig)

		# color bar figure
		fig_cb, ax_cb = plt.subplots(figsize=(0.5, 4))  # Tall vertical colorbar
		fig_cb.subplots_adjust(left=0.45, right=0.8, top=0.95, bottom=0.05)
		cbar = fig_cb.colorbar(tracts, cax=ax_cb)
		ax_cb.tick_params(left=False, labelleft=False, right=False)

		# render colorbar image
		canvas_cb = FigureCanvas(fig_cb)
		canvas_cb.draw()
		w_cb, h_cb = fig_cb.get_size_inches() * fig_cb.get_dpi()
		colorbar_img = np.frombuffer(canvas_cb.tostring_rgb(), dtype='uint8').reshape(int(h_cb), int(w_cb), 3)
		plt.close(fig_cb)

		return quiver_img, colorbar_img
