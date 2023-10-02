
from mpi4py import MPI
import interpol as intr
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import numpy.fft as t
import time


from mpl_toolkits.axes_grid1 import make_axes_locatable

AXIS_X_1D = 0

AXIS_Y_2D = 0
AXIS_X_2D = 1

AXIS_Z_3D = 0
AXIS_Y_3D = 1
AXIS_X_3D = 2

DIR_BITS = 5
VAR_BITS = DIR_BITS + 4

SXX = 1<<DIR_BITS
SYY = 2<<DIR_BITS
SXY = 3<<DIR_BITS

UX = 4<<DIR_BITS
UY = 5<<DIR_BITS
P  = 6<<DIR_BITS

FLUID_MODEL   = 0
ELASTIC_MODEL = 1

# 					  y  x
#NEIGHBOUR_OFFSET = [(-1,-1), (-1,0), (-1,1), (0,-1), (0, 1), (1,-1), (1,0), (1,1)]

class Extractor:
	"""class handling overlaps extraction and exchange"""
	def __init__(self, N_global, N, d_c, overlap, base_overlap, ds, dt, t_end, num_domains, model_map, rank, comm, comm_frequency, interpolation):
		self
		super(Extractor, self).__init__()
		
		if rank >= num_domains or rank < 0:
			raise Exception("Domain number out of range")

		self.single_res = True
		self.single_dt  = True

		if (len(N) != 1) and (len(N) != num_domains):
			raise Exception("Either single dimensions or dimensions per domain have to be specified")
		
		if not self.single_res and (len(overlap) != num_domains):
			raise Exception("overap size has to be specified per domain")
		
		if not self.single_res and (len(overlap) != num_domains):
			raise Exception("spatial resolution has to be specified per domain")	

		self.N_global            = N_global
		self.d_cs 	= d_c
		self.d_c 	= d_c[rank]
		self.offset_g = self.d_c[0]
		self.N_base = (d_c[rank][1][0] - d_c[rank][0][0], d_c[rank][1][1] - d_c[rank][0][1])

		self.ds = ds[rank]
		self.N  = N[rank]
		self.overlaps = overlap
		self.overlap = overlap[rank]
		self.base_overlap = base_overlap
		self.N_i_o   = (N[rank][0] + 2*overlap[rank][0], N[rank][1] + 2*overlap[rank][1])
		self.dt = dt[rank]
		self.model = model_map[rank]
		self.comm_freq = comm_frequency[rank]

		if self.comm_freq > 1:
			bell_x = self.get_bell(self.N[1] + 2*self.overlap[1], int(self.overlap[1]/2))
			bell_x = bell_x.reshape((1, self.N[1] + 2*self.overlap[1]))

			bell_y = self.get_bell(self.N[0] + 2*self.overlap[0], int(self.overlap[0]/2))
			bell_y = bell_y.reshape((self.N[0] + 2*self.overlap[0], 1))

		else:
			bell_x = self.get_bell(self.N[1] + 2*self.overlap[1], int(self.overlap[1]))
			bell_x = bell_x.reshape((1, self.N[1] + 2*self.overlap[1]))

			bell_y = self.get_bell(self.N[0] + 2*self.overlap[0], int(self.overlap[0]))
			bell_y = bell_y.reshape((self.N[0] + 2*self.overlap[0], 1))

		self.on_base_res = False
		if 	self.N_base[0] == self.N[0] and self.N_base[1] == self.N[1]:
			self.on_base_res = True

		self.res_coeff = (self.N[0] / self.N_base[0], self.N[1] / self.N_base[1]) 

		self.num_domains = num_domains
		self.rank 		 = rank
		self.comm 		 = comm

		self.bell   = np.matmul(bell_y, bell_x)
		self.neighbours_dir_to_rank_map = []
		self.dir_neighbour_N          = []
		self.dir_neighbour_overlap    = []
		self.dir_neighbour_dt	      = []
		self.dir_send_neighbour_model = []
		self.dir_recv_neighbour_model = []
		self.dir_neighbour_comm_freq  = []
		self.dir_neighbour_res_coeffs = []

		self.neighbour_res_coeffs = []

		self.unique_resample_coeffs        = dict()
		self.unique_spectral_uy_correction = dict()
		self.unique_spectral_ux_correction = dict()
		self.unique_uy_correction          = dict()
		self.unique_ux_correction          = dict()

		self.interpolation = interpolation

		if self.interpolation == 1 or self.interpolation == 7:
			self.method = 'nearest'

		if self.interpolation == 2 or self.interpolation == 8:
			self.method = 'linear'

		if self.interpolation == 3 or self.interpolation == 9:
			self.order = 2

		if self.interpolation == 4 or self.interpolation == 10:
			self.order = 3

		if self.interpolation == 5 or self.interpolation == 11:
			self.order = 4
		
		if self.interpolation == 6 or self.interpolation == 12:
			self.order = 5
		
		for i in range(self.num_domains):
			if self.N != N[i]:
				self.single_res = False

			if self.dt != dt[i]:
				self.single_dt = False
		
		if self.rank == 0:
			print("Extractor: Basic initialization done")

		for i in range(self.num_domains):
			N_base = (d_c[i][1][0] - d_c[i][0][0], d_c[i][1][1] - d_c[i][0][1])
			self.neighbour_res_coeffs.append((N[i][0] / N_base[0], N[i][1] / N_base[1]))

		self.detect_neighbours()

		if self.rank == 0:
			print("Extractor: Neighbours detected")

		for entry in self.recvneighbours:

			neighbour_index = entry[0]
			
			self.neighbours_dir_to_rank_map.append(neighbour_index)

			self.dir_recv_neighbour_model.append(model_map[neighbour_index])
		
		for entry in self.sendneighbours:
 
			neighbour_index = entry[0]
			
			self.neighbours_dir_to_rank_map.append(neighbour_index)

			self.dir_send_neighbour_model.append(model_map[neighbour_index])
			self.dir_neighbour_comm_freq.append(comm_frequency[neighbour_index])

			if self.single_res:
				N_base = (d_c[0][1][0] - d_c[0][0][0], d_c[0][1][1] - d_c[0][0][1])
				self.dir_neighbour_res_coeffs.append((N[0][0] / N_base[0], N[0][1] / N_base[1]))
			else:
				N_base = (d_c[neighbour_index][1][0] - d_c[neighbour_index][0][0], d_c[neighbour_index][1][1] - d_c[neighbour_index][0][1])
				self.dir_neighbour_res_coeffs.append((N[neighbour_index][0] / N_base[0], N[neighbour_index][1] / N_base[1]))
			
			if self.single_res:
				self.dir_neighbour_N.append(N[0])
				self.dir_neighbour_overlap.append(overlap[0])
			else:
				self.dir_neighbour_N.append(N[neighbour_index])
				self.dir_neighbour_overlap.append(overlap[neighbour_index])
			
			if not self.single_res:	
				self.unique_resample_coeffs       [self.dir_neighbour_res_coeffs[-1]] = (self.dir_neighbour_res_coeffs[-1][0] / self.res_coeff[0], self.dir_neighbour_res_coeffs[-1][1] / self.res_coeff[1])
				self.unique_spectral_uy_correction[self.dir_neighbour_res_coeffs[-1]] = self.get_spectral_shift(ds[neighbour_index][0]/2 - self.ds[0]/2, 0)
				self.unique_spectral_ux_correction[self.dir_neighbour_res_coeffs[-1]] = self.get_spectral_shift(ds[neighbour_index][1]/2 - self.ds[1]/2, 1)
				self.unique_uy_correction         [self.dir_neighbour_res_coeffs[-1]] = ds[neighbour_index][0]/2 - self.ds[0]/2
				self.unique_ux_correction         [self.dir_neighbour_res_coeffs[-1]] = ds[neighbour_index][1]/2 - self.ds[1]/2

			else:
				self.unique_resample_coeffs       [self.dir_neighbour_res_coeffs[0]] = (self.dir_neighbour_res_coeffs[-1][0] / self.res_coeff[0], self.dir_neighbour_res_coeffs[-1][1] / self.res_coeff[1])
				self.unique_spectral_uy_correction[self.dir_neighbour_res_coeffs[0]] = self.get_spectral_shift( 0, 0)
				self.unique_spectral_ux_correction[self.dir_neighbour_res_coeffs[0]] = self.get_spectral_shift( 0, 1)
				self.unique_uy_correction         [self.dir_neighbour_res_coeffs[0]] = 0
				self.unique_ux_correction         [self.dir_neighbour_res_coeffs[0]] = 0

			# if not self.single_dt:
			self.dir_neighbour_dt.append(dt[neighbour_index])

		if self.rank == 0:
			print("Extractor: Neighbours related data setup")

		self.iterations = int(t_end / self.dt)
		rem = t_end - (self.iterations * self.dt)
		if not np.isclose(0.0, rem):
			self.iterations = self.iterations + 1

		self.neigbour_a_step_index_todo = {}
		self.neigbour_b_step_index_todo = {}


		self.a_step_index_todo = []
		self.b_step_index_todo = []
		
		unique_list = []
		# traverse for all elements
		for x in self.dir_neighbour_dt:
			if x not in unique_list:
				unique_list.append(x)
		
		for i in range(0, self.iterations):
			self.b_step_index_todo.append({})
			self.a_step_index_todo.append({})
		self.b_step_index_todo.append({})

		for neighbour_dt in unique_list:
			neighbour_iterations = int(t_end / neighbour_dt)
			rem = t_end - (neighbour_iterations * neighbour_dt)
			if not np.isclose(0.0, rem):
				neighbour_iterations = neighbour_iterations + 1

			b_step_index_todo = []
			my_b_step = 0;
			my_timestamp = -(self.dt/2)

			for neighbor_step in range (1, neighbour_iterations+1):
				nodge = np.min([self.dt, neighbour_dt])/10.0
				neighbor_timestamp = neighbour_dt*neighbor_step
				if not np.isclose(0.0, rem) and neighbor_step == neighbour_iterations:
					neighbor_timestamp = t_end
				while (my_b_step <= self.iterations) and (my_timestamp < (neighbor_timestamp-nodge)):
					rho_gen_by = my_b_step
					my_b_step += 1;
					my_timestamp = self.dt*my_b_step -(self.dt/2)
					if my_timestamp > (t_end - (self.dt/2)):
						my_timestamp = self.dt*(my_b_step-1) + (t_end - self.dt*(my_b_step-1))/2
					b_step_index_todo.append([])
				if rho_gen_by == my_b_step-1:
					b_step_index_todo[my_b_step-1].append((neighbor_timestamp, neighbor_step))
					self.b_step_index_todo[my_b_step-1][neighbor_timestamp] = neighbor_timestamp



			a_step_index_todo = []
			my_a_step = 0;
			my_timestamp = 0
			for neighbor_step in range (1, neighbour_iterations+1):
				nodge = np.min([self.dt, neighbour_dt])/10.0	
				neighbor_timestamp = neighbour_dt*neighbor_step-(neighbour_dt/2)
				if not np.isclose(0.0, rem) and neighbor_step == neighbour_iterations:
					neighbor_timestamp = neighbour_dt*(neighbor_step - 1) + (t_end - neighbour_dt*(neighbor_step - 1))/2.0

				
				while (my_a_step < self.iterations) and (my_timestamp < (neighbor_timestamp+nodge)) :
					u_gen_by = my_a_step
					my_a_step += 1;
					my_timestamp = self.dt*my_a_step
					if my_timestamp > t_end:
						my_timestamp = t_end
					a_step_index_todo.append([])
				if u_gen_by == my_a_step-1:
					a_step_index_todo[my_a_step-1].append((neighbor_timestamp, neighbor_step))
					self.a_step_index_todo[my_a_step-1][neighbor_timestamp] = neighbor_timestamp

			


			
			for i in range(0,  self.iterations      - len(a_step_index_todo)):
				a_step_index_todo.append([])
			for i in range(0, (self.iterations + 1) - len(b_step_index_todo)):
				b_step_index_todo.append([])

			self.neigbour_a_step_index_todo[neighbour_dt] = a_step_index_todo
			self.neigbour_b_step_index_todo[neighbour_dt] = b_step_index_todo

		self.a_step_max_slots = {} 
		for neighbour_dt, todo_lists in self.neigbour_a_step_index_todo.items():
			max_steps_todo = 0
			for todo_list in todo_lists:
				if len(todo_list) > max_steps_todo:
					max_steps_todo = len(todo_list)
			self.a_step_max_slots[neighbour_dt] = max_steps_todo

		self.b_step_max_slots = {}
		for neighbour_dt, todo_lists in self.neigbour_b_step_index_todo.items():
			max_steps_todo = 0
			for todo_list in todo_lists:
				if len(todo_list) > max_steps_todo:
					max_steps_todo = len(todo_list)
			self.b_step_max_slots[neighbour_dt] = max_steps_todo

		if self.rank == 0:
			print("Extractor: Non-uniform timestep setup")

		self.ux_overlap_recv_buf  = []
		self.uy_overlap_recv_buf  = []
		self.rho_overlap_recv_buf   = []
		self.sxx_overlap_recv_buf = []
		self.syy_overlap_recv_buf = []
		self.sxy_overlap_recv_buf = []

		self.ux_overlap_send_buf  = []
		self.uy_overlap_send_buf  = []
		self.rho_overlap_send_buf   = []
		self.sxx_overlap_send_buf = []
		self.syy_overlap_send_buf = []
		self.sxy_overlap_send_buf = []

		self.ux_overlap_send_buf_rq  = []
		self.uy_overlap_send_buf_rq  = []
		self.rho_overlap_send_buf_rq   = []
		self.sxx_overlap_send_buf_rq = []
		self.syy_overlap_send_buf_rq = []
		self.sxy_overlap_send_buf_rq = []

		self.velocity_send_buffers_num   = [] 
		self.density_send_buffers_num   = []
		self.velocity_send_buffers_index = []
		self.density_send_buffers_index = []

		for meighbour_index, entry in enumerate(self.sendneighbours):
			# Y offset 0 == collumn exchange else corner or row exchange
			neighbour_rank        = entry[0]
			send_corners = entry[1]

			n_N 	  = self.dir_neighbour_N[meighbour_index]
			n_overlap = self.dir_neighbour_overlap[meighbour_index]
			n_dt 	  = self.dir_neighbour_dt[meighbour_index]	

			n_size_y = send_corners[1][0] - send_corners[0][0]
			n_size_x = send_corners[1][1] - send_corners[0][1]

			n_size_y = int(n_size_y * self.dir_neighbour_res_coeffs[meighbour_index][0])
			n_size_x = int(n_size_x * self.dir_neighbour_res_coeffs[meighbour_index][1])

			velocity_send_buffers = self.a_step_max_slots[n_dt]
			density_send_buffers  = self.b_step_max_slots[n_dt]

			self.velocity_send_buffers_num.append(velocity_send_buffers)
			self.density_send_buffers_num.append(density_send_buffers)
			
			self.velocity_send_buffers_index.append(0)
			self.density_send_buffers_index.append(0)
			
			ux_overlap_send_buf = []
			uy_overlap_send_buf = []

			ux_overlap_send_buf_rq = []
			uy_overlap_send_buf_rq = []

			for i in range(0, velocity_send_buffers):
				ux_overlap_send_buf.append(np.zeros((n_size_y, n_size_x)))
				uy_overlap_send_buf.append(np.zeros((n_size_y, n_size_x)))
				
				ux_overlap_send_buf_rq.append(MPI.Request())
				uy_overlap_send_buf_rq.append(MPI.Request())

			self.ux_overlap_send_buf.append(ux_overlap_send_buf)
			self.uy_overlap_send_buf.append(uy_overlap_send_buf)

			self.ux_overlap_send_buf_rq.append(ux_overlap_send_buf_rq)
			self.uy_overlap_send_buf_rq.append(uy_overlap_send_buf_rq)

			rho_overlap_send_buf   = []
			sxx_overlap_send_buf = []
			syy_overlap_send_buf = []
			sxy_overlap_send_buf = []

			rho_overlap_send_buf_rq  = []
			sxx_overlap_send_buf_rq = []
			syy_overlap_send_buf_rq = []
			sxy_overlap_send_buf_rq = []

			for i in range(0, density_send_buffers):
				rho_overlap_send_buf.append(np.zeros((n_size_y, n_size_x)))
				sxx_overlap_send_buf.append(np.zeros((n_size_y, n_size_x)))
				syy_overlap_send_buf.append(np.zeros((n_size_y, n_size_x)))
				sxy_overlap_send_buf.append(np.zeros((n_size_y, n_size_x)))

				rho_overlap_send_buf_rq.append(MPI.Request())
				sxx_overlap_send_buf_rq.append(MPI.Request())
				syy_overlap_send_buf_rq.append(MPI.Request())	
				sxy_overlap_send_buf_rq.append(MPI.Request())
			
			self.rho_overlap_send_buf.append(rho_overlap_send_buf)
			self.sxx_overlap_send_buf.append(sxx_overlap_send_buf)
			self.syy_overlap_send_buf.append(syy_overlap_send_buf)
			self.sxy_overlap_send_buf.append(sxy_overlap_send_buf)

			self.rho_overlap_send_buf_rq.append(rho_overlap_send_buf_rq)
			self.sxx_overlap_send_buf_rq.append(sxx_overlap_send_buf_rq)
			self.syy_overlap_send_buf_rq.append(syy_overlap_send_buf_rq)
			self.sxy_overlap_send_buf_rq.append(sxy_overlap_send_buf_rq)

		if self.rank == 0:
			print("Extractor: Send buffers alocated")

		for entry in self.recvneighbours:
			recv_corners = entry[1]

			size_y = recv_corners[1][0] - recv_corners[0][0]
			size_x = recv_corners[1][1] - recv_corners[0][1]

			self.rho_overlap_recv_buf.append(np.zeros((size_y, size_x)))

			self.ux_overlap_recv_buf.append(np.zeros((size_y, size_x)))
			self.uy_overlap_recv_buf.append(np.zeros((size_y, size_x)))

			self.sxx_overlap_recv_buf.append(np.zeros((size_y, size_x)))
			self.syy_overlap_recv_buf.append(np.zeros((size_y, size_x)))
			self.sxy_overlap_recv_buf.append(np.zeros((size_y, size_x)))

		if self.rank == 0:
			print("Extractor: Receive buffers alocated")

		self.global_points_x = np.arange(self.offset_g[1] - (self.overlap[1] / self.res_coeff[1]), self.offset_g[1] + self.N_base[1] + (self.overlap[1] / self.res_coeff[1]), 1 / self.res_coeff[1]) 
		self.global_points_y = np.arange(self.offset_g[0] - (self.overlap[0] / self.res_coeff[0]), self.offset_g[0] + self.N_base[0] + (self.overlap[0] / self.res_coeff[0]), 1 / self.res_coeff[0]) 

		self.global_points_x_sgx = self.global_points_x + 1 / (2*self.res_coeff[1]);
		self.global_points_y_sgy = self.global_points_y + 1 / (2*self.res_coeff[0]);

		self.global_points_x = self.global_points_x % self.N_global[1]
		self.global_points_y = self.global_points_y % self.N_global[0]

		self.global_points_x_sgx = self.global_points_x_sgx % self.N_global[1]
		self.global_points_y_sgy = self.global_points_y_sgy % self.N_global[0]

		self.X, self.Y         = np.meshgrid(self.global_points_x,     self.global_points_y)
		self.X_SGX, self.Y_SGY = np.meshgrid(self.global_points_x_sgx, self.global_points_y_sgy)

		self.X = self.X.flatten()
		self.Y = self.Y.flatten()

		self.X_SGX = self.X_SGX.flatten()
		self.Y_SGY = self.Y_SGY.flatten()

		self.sendbuf = np.empty(self.N_base)
		self.recvbuf = None
		
		if self.rank == 0:
			print("Extractor: Index arrays setup")

		self.sendcounts = np.array(self.comm.gather(len(self.sendbuf.flatten()), 0))


		self.sendcounts_scan = None
		if rank == 0:
			self.sendcounts_scan = [0]
			cum_count = 0
			for count in self.sendcounts:
				cum_count += count
				self.sendcounts_scan.append(cum_count)

			self.recvbuf = np.empty(sum(self.sendcounts))
	
	def detect_neighbours(self):
		self.sendneighbours = []
		for rank in range(self.num_domains):
			neighbour_d_c = self.d_cs[rank]
			res_coeff     = self.neighbour_res_coeffs[rank]

			y1 = neighbour_d_c[0][0]
			x1 = neighbour_d_c[0][1]
			y2 = neighbour_d_c[1][0]
			x2 = neighbour_d_c[1][1]

			my_overlap_y = int(self.overlaps[rank][0] / res_coeff[0])
			my_overlap_x = int(self.overlaps[rank][1] / res_coeff[1])


			curr_y1 = (y1 - my_overlap_y)
			curr_x1 = (x1 - my_overlap_x)
			curr_y2 = y1
			curr_x2 = x1


			if curr_y1 >= self.N_global[0]:
				curr_y1 = curr_y1 % self.N_global[0] 
				curr_y2 = curr_y2 % self.N_global[0]

			if curr_x1 >= self.N_global[1]:
				curr_x1 = curr_x1 % self.N_global[1] 
				curr_x2 = curr_x2 % self.N_global[1]

			if curr_y1 < 0:
				curr_y1 = curr_y1 % self.N_global[0] 
				curr_y2 = self.N_global[0]

			if curr_x1 < 0:
				curr_x1 = curr_x1 % self.N_global[1] 
				curr_x2 = self.N_global[1]

			res = self.detect_intersection(self.d_c, [(curr_y1, curr_x1),(curr_y2, curr_x2)])
			if res != None:
				local_y1 = int((res[0][0] - self.d_c[0][0] + my_overlap_y) * res_coeff[0])
				local_x1 = int((res[0][1] - self.d_c[0][1] + my_overlap_x) * res_coeff[1])
				local_y2 = int((res[1][0] - self.d_c[0][0] + my_overlap_y) * res_coeff[0])
				local_x2 = int((res[1][1] - self.d_c[0][1] + my_overlap_x) * res_coeff[1])

				self.sendneighbours.append([rank, [(local_y1, local_x1), (local_y2, local_x2)], 0])
			###########################################################################
			curr_y1 = (y1 - my_overlap_y)
			curr_x1 = x1 
			curr_y2 = y1 
			curr_x2 = x2 

			if curr_y1 >= self.N_global[0]:
				curr_y1 = curr_y1 % self.N_global[0] 
				curr_y2 = curr_y2 % self.N_global[0]

			if curr_x1 >= self.N_global[1]:
				curr_x1 = curr_x1 % self.N_global[1] 
				curr_x2 = curr_x2 % self.N_global[1]

			if curr_y1 < 0:
				curr_y1 = curr_y1 % self.N_global[0] 
				curr_y2 = self.N_global[0]

			if curr_x1 < 0:
				curr_x1 = curr_x1 % self.N_global[1] 
				curr_x2 = self.N_global[1]

			res = self.detect_intersection(self.d_c, [(curr_y1, curr_x1),(curr_y2, curr_x2)])
			if res != None:
				local_y1 = int((res[0][0] - self.d_c[0][0] + my_overlap_y) * res_coeff[0])
				local_x1 = int((res[0][1] - self.d_c[0][1] + my_overlap_x) * res_coeff[1])
				local_y2 = int((res[1][0] - self.d_c[0][0] + my_overlap_y) * res_coeff[0])
				local_x2 = int((res[1][1] - self.d_c[0][1] + my_overlap_x) * res_coeff[1])

				self.sendneighbours.append([rank, [(local_y1, local_x1), (local_y2, local_x2)], 1])
			###########################################################################
			curr_y1 = (y1 - my_overlap_y)
			curr_x1 = x2 
			curr_y2 = y1 
			curr_x2 = (x2 + my_overlap_x)

			if curr_y1 >= self.N_global[0]:
				curr_y1 = curr_y1 % self.N_global[0] 
				curr_y2 = curr_y2 % self.N_global[0]

			if curr_x1 >= self.N_global[1]:
				curr_x1 = curr_x1 % self.N_global[1] 
				curr_x2 = curr_x2 % self.N_global[1]

			if curr_y1 < 0:
				curr_y1 = curr_y1 % self.N_global[0] 
				curr_y2 = self.N_global[0]

			if curr_x1 < 0:
				curr_x1 = curr_x1 % self.N_global[1] 
				curr_x2 = self.N_global[1]

			res = self.detect_intersection(self.d_c, [(curr_y1, curr_x1),(curr_y2, curr_x2)])
			if res != None:
				local_y1 = int((res[0][0] - self.d_c[0][0] + my_overlap_y) * res_coeff[0])
				local_x1 = int((res[0][1] - self.d_c[0][1] + my_overlap_x) * res_coeff[1])
				local_y2 = int((res[1][0] - self.d_c[0][0] + my_overlap_y) * res_coeff[0])
				local_x2 = int((res[1][1] - self.d_c[0][1] + my_overlap_x) * res_coeff[1])

				self.sendneighbours.append([rank, [(local_y1, local_x1), (local_y2, local_x2)], 2])
			########################################################################### 
			curr_y1 = y1 
			curr_x1 = (x1 - my_overlap_x) 
			curr_y2 = y2
			curr_x2 = x1 

			if curr_y1 >= self.N_global[0]:
				curr_y1 = curr_y1 % self.N_global[0] 
				curr_y2 = curr_y2 % self.N_global[0]

			if curr_x1 >= self.N_global[1]:
				curr_x1 = curr_x1 % self.N_global[1] 
				curr_x2 = curr_x2 % self.N_global[1]

			if curr_y1 < 0:
				curr_y1 = curr_y1 % self.N_global[0] 
				curr_y2 = self.N_global[0]

			if curr_x1 < 0:
				curr_x1 = curr_x1 % self.N_global[1] 
				curr_x2 = self.N_global[1]
			

			res = self.detect_intersection(self.d_c, [(curr_y1, curr_x1),(curr_y2, curr_x2)])
			if res != None:
				local_y1 = int((res[0][0] - self.d_c[0][0] + my_overlap_y) * res_coeff[0])
				local_x1 = int((res[0][1] - self.d_c[0][1] + my_overlap_x) * res_coeff[1])
				local_y2 = int((res[1][0] - self.d_c[0][0] + my_overlap_y) * res_coeff[0])
				local_x2 = int((res[1][1] - self.d_c[0][1] + my_overlap_x) * res_coeff[1])

				self.sendneighbours.append([rank, [(local_y1, local_x1), (local_y2, local_x2)], 3])
			###########################################################################
			curr_y1 = y1
			curr_x1 = x2
			curr_y2 = y2
			curr_x2 = (x2 + my_overlap_x)

			if curr_y1 >= self.N_global[0]:
				curr_y1 = curr_y1 % self.N_global[0] 
				curr_y2 = curr_y2 % self.N_global[0]

			if curr_x1 >= self.N_global[1]:
				curr_x1 = curr_x1 % self.N_global[1] 
				curr_x2 = curr_x2 % self.N_global[1]

			if curr_y1 < 0:
				curr_y1 = curr_y1 % self.N_global[0] 
				curr_y2 = self.N_global[0]

			if curr_x1 < 0:
				curr_x1 = curr_x1 % self.N_global[1] 
				curr_x2 = self.N_global[1]

			res = self.detect_intersection(self.d_c, [(curr_y1, curr_x1),(curr_y2, curr_x2)])
			if res != None:
				local_y1 = int((res[0][0] - self.d_c[0][0] + my_overlap_y) * res_coeff[0])
				local_x1 = int((res[0][1] - self.d_c[0][1] + my_overlap_x) * res_coeff[1])
				local_y2 = int((res[1][0] - self.d_c[0][0] + my_overlap_y) * res_coeff[0])
				local_x2 = int((res[1][1] - self.d_c[0][1] + my_overlap_x) * res_coeff[1])

				self.sendneighbours.append([rank, [(local_y1, local_x1), (local_y2, local_x2)], 4])
			###########################################################################
			curr_y1 = y2 
			curr_x1 = (x1 - my_overlap_x) 
			curr_y2 = (y2 + my_overlap_y) 
			curr_x2 = x1 

			if curr_y1 >= self.N_global[0]:
				curr_y1 = curr_y1 % self.N_global[0] 
				curr_y2 = curr_y2 % self.N_global[0]

			if curr_x1 >= self.N_global[1]:
				curr_x1 = curr_x1 % self.N_global[1] 
				curr_x2 = curr_x2 % self.N_global[1]

			if curr_y1 < 0:
				curr_y1 = curr_y1 % self.N_global[0] 
				curr_y2 = self.N_global[0]

			if curr_x1 < 0:
				curr_x1 = curr_x1 % self.N_global[1] 
				curr_x2 = self.N_global[1]

			res = self.detect_intersection(self.d_c, [(curr_y1, curr_x1),(curr_y2, curr_x2)])
			if res != None:
				local_y1 = int((res[0][0] - self.d_c[0][0] + my_overlap_y) * res_coeff[0])
				local_x1 = int((res[0][1] - self.d_c[0][1] + my_overlap_x) * res_coeff[1])
				local_y2 = int((res[1][0] - self.d_c[0][0] + my_overlap_y) * res_coeff[0])
				local_x2 = int((res[1][1] - self.d_c[0][1] + my_overlap_x) * res_coeff[1])

				self.sendneighbours.append([rank, [(local_y1, local_x1), (local_y2, local_x2)], 5])
			###########################################################################
			curr_y1 = y2 
			curr_x1 = x1 
			curr_y2 = (y2 + my_overlap_y)
			curr_x2 = x2

			if curr_y1 >= self.N_global[0]:
				curr_y1 = curr_y1 % self.N_global[0] 
				curr_y2 = curr_y2 % self.N_global[0]

			if curr_x1 >= self.N_global[1]:
				curr_x1 = curr_x1 % self.N_global[1] 
				curr_x2 = curr_x2 % self.N_global[1]

			if curr_y1 < 0:
				curr_y1 = curr_y1 % self.N_global[0] 
				curr_y2 = self.N_global[0]

			if curr_x1 < 0:
				curr_x1 = curr_x1 % self.N_global[1] 
				curr_x2 = self.N_global[1]

			res = self.detect_intersection(self.d_c, [(curr_y1, curr_x1),(curr_y2, curr_x2)])
			if res != None:
				local_y1 = int((res[0][0] - self.d_c[0][0] + my_overlap_y) * res_coeff[0])
				local_x1 = int((res[0][1] - self.d_c[0][1] + my_overlap_x) * res_coeff[1])
				local_y2 = int((res[1][0] - self.d_c[0][0] + my_overlap_y) * res_coeff[0])
				local_x2 = int((res[1][1] - self.d_c[0][1] + my_overlap_x) * res_coeff[1])

				self.sendneighbours.append([rank, [(local_y1, local_x1), (local_y2, local_x2)], 6])
			###########################################################################
			curr_y1 = y2 
			curr_x1 = x2 
			curr_y2 = (y2 + my_overlap_y) 
			curr_x2 = (x2 + my_overlap_x) 

			if curr_y1 >= self.N_global[0]:
				curr_y1 = curr_y1 % self.N_global[0] 
				curr_y2 = curr_y2 % self.N_global[0]

			if curr_x1 >= self.N_global[1]:
				curr_x1 = curr_x1 % self.N_global[1] 
				curr_x2 = curr_x2 % self.N_global[1]

			if curr_y1 < 0:
				curr_y1 = curr_y1 % self.N_global[0] 
				curr_y2 = self.N_global[0]

			if curr_x1 < 0:
				curr_x1 = curr_x1 % self.N_global[1] 
				curr_x2 = self.N_global[1]

			res = self.detect_intersection(self.d_c, [(curr_y1, curr_x1),(curr_y2, curr_x2)])
			if res != None:
				local_y1 = int((res[0][0] - self.d_c[0][0] + my_overlap_y) * res_coeff[0])
				local_x1 = int((res[0][1] - self.d_c[0][1] + my_overlap_x) * res_coeff[1])
				local_y2 = int((res[1][0] - self.d_c[0][0] + my_overlap_y) * res_coeff[0])
				local_x2 = int((res[1][1] - self.d_c[0][1] + my_overlap_x) * res_coeff[1])

				self.sendneighbours.append([rank, [(local_y1, local_x1), (local_y2, local_x2)], 7])

####################################################################################################
		self.recvneighbours = []
		for rank in range(self.num_domains):
			my_res_coeff     = self.neighbour_res_coeffs[self.rank]
			y1 = int(self.d_c[0][0] * my_res_coeff[0])
			x1 = int(self.d_c[0][1] * my_res_coeff[1])
			y2 = int(self.d_c[1][0] * my_res_coeff[0])
			x2 = int(self.d_c[1][1] * my_res_coeff[1])

			neigh_y1 = int(self.d_cs[rank][0][0] * my_res_coeff[0])
			neigh_x1 = int(self.d_cs[rank][0][1] * my_res_coeff[1])
			neigh_y2 = int(self.d_cs[rank][1][0] * my_res_coeff[0])
			neigh_x2 = int(self.d_cs[rank][1][1] * my_res_coeff[1])

			neigh_dc_mod = [(neigh_y1, neigh_x1), (neigh_y2, neigh_x2)]


			N_global_mod = [int(self.N_global[0] * my_res_coeff[0]), int(self.N_global[1] * my_res_coeff[1])]
			
			overlap_y = self.overlap[0]
			overlap_x = self.overlap[1]

			curr_y1 = (y1 - overlap_y)
			curr_x1 = (x1 - overlap_x)
			curr_y2 = y1
			curr_x2 = x1

			if curr_y1 >= N_global_mod[0]:
				curr_y1 = curr_y1 % N_global_mod[0] 
				curr_y2 = curr_y2 % N_global_mod[0]

			if curr_x1 >= N_global_mod[1]:
				curr_x1 = curr_x1 % N_global_mod[1] 
				curr_x2 = curr_x2 % N_global_mod[1]

			if curr_y1 < 0:
				curr_y1 = curr_y1 % N_global_mod[0] 
				curr_y2 =           N_global_mod[0]

			if curr_x1 < 0:
				curr_x1 = curr_x1 % N_global_mod[1] 
				curr_x2 =           N_global_mod[1]

			res = self.detect_intersection(neigh_dc_mod, [(curr_y1, curr_x1),(curr_y2, curr_x2)])
			if res != None:
				diff_y1 = res[0][0] - curr_y1
				diff_x1 = res[0][1] - curr_x1
				diff_y2 = curr_y2 - res[1][0] 
				diff_x2 = curr_x2 - res[1][1] 

				local_y1 = 0
				local_x1 = 0
				local_y2 = overlap_y
				local_x2 = overlap_x

				self.recvneighbours.append([rank, [(local_y1 + diff_y1, local_x1 + diff_x1), (local_y2 - diff_y2, local_x2 - diff_x2)], 0])
			###########################################################################
			curr_y1 = (y1 - overlap_y)
			curr_x1 = x1 
			curr_y2 = y1 
			curr_x2 = x2 

			if curr_y1 >= N_global_mod[0]:
				curr_y1 = curr_y1 % N_global_mod[0] 
				curr_y2 = curr_y2 % N_global_mod[0]

			if curr_x1 >= N_global_mod[1]:
				curr_x1 = curr_x1 % N_global_mod[1] 
				curr_x2 = curr_x2 % N_global_mod[1]

			if curr_y1 < 0:
				curr_y1 = curr_y1 % N_global_mod[0] 
				curr_y2 =           N_global_mod[0]

			if curr_x1 < 0:
				curr_x1 = curr_x1 % N_global_mod[1] 
				curr_x2 =           N_global_mod[1]

			res = self.detect_intersection(neigh_dc_mod, [(curr_y1, curr_x1),(curr_y2, curr_x2)])
			if res != None:
				diff_y1 = res[0][0] - curr_y1
				diff_x1 = res[0][1] - curr_x1
				diff_y2 = curr_y2 - res[1][0] 
				diff_x2 = curr_x2 - res[1][1] 

				local_y1 = 0
				local_x1 = overlap_x
				local_y2 = overlap_y
				local_x2 = self.N_i_o[1] - overlap_x

				self.recvneighbours.append([rank, [(local_y1 + diff_y1, local_x1 + diff_x1), (local_y2 - diff_y2, local_x2 - diff_x2)], 1])
			###########################################################################
			curr_y1 = (y1 - overlap_y)
			curr_x1 = x2 
			curr_y2 = y1 
			curr_x2 = (x2 + overlap_x)

			if curr_y1 >= N_global_mod[0]:
				curr_y1 = curr_y1 % N_global_mod[0] 
				curr_y2 = curr_y2 % N_global_mod[0]

			if curr_x1 >= N_global_mod[1]:
				curr_x1 = curr_x1 % N_global_mod[1] 
				curr_x2 = curr_x2 % N_global_mod[1]

			if curr_y1 < 0:
				curr_y1 = curr_y1 % N_global_mod[0] 
				curr_y2 =           N_global_mod[0]

			if curr_x1 < 0:
				curr_x1 = curr_x1 % N_global_mod[1] 
				curr_x2 =           N_global_mod[1]

			res = self.detect_intersection(neigh_dc_mod, [(curr_y1, curr_x1),(curr_y2, curr_x2)])
			if res != None:
				diff_y1 = res[0][0] - curr_y1
				diff_x1 = res[0][1] - curr_x1
				diff_y2 = curr_y2 - res[1][0] 
				diff_x2 = curr_x2 - res[1][1] 

				local_y1 = 0
				local_x1 = self.N_i_o[1] - overlap_x
				local_y2 = overlap_y
				local_x2 = self.N_i_o[1]
				
				self.recvneighbours.append([rank, [(local_y1 + diff_y1, local_x1 + diff_x1), (local_y2 - diff_y2, local_x2 - diff_x2)], 2])
			########################################################################### 
			curr_y1 = y1 
			curr_x1 = (x1 - overlap_x) 
			curr_y2 = y2
			curr_x2 = x1 

			if curr_y1 >= N_global_mod[0]:
				curr_y1 = curr_y1 % N_global_mod[0] 
				curr_y2 = curr_y2 % N_global_mod[0]

			if curr_x1 >= N_global_mod[1]:
				curr_x1 = curr_x1 % N_global_mod[1] 
				curr_x2 = curr_x2 % N_global_mod[1]

			if curr_y1 < 0:
				curr_y1 = curr_y1 % N_global_mod[0] 
				curr_y2 =           N_global_mod[0]

			if curr_x1 < 0:
				curr_x1 = curr_x1 % N_global_mod[1] 
				curr_x2 =           N_global_mod[1]
			

			res = self.detect_intersection(neigh_dc_mod, [(curr_y1, curr_x1),(curr_y2, curr_x2)])
			if res != None:
				diff_y1 = res[0][0] - curr_y1
				diff_x1 = res[0][1] - curr_x1
				diff_y2 = curr_y2 - res[1][0] 
				diff_x2 = curr_x2 - res[1][1] 

				local_y1 = overlap_y
				local_x1 = 0
				local_y2 = self.N_i_o[0] - overlap_y
				local_x2 = overlap_x

				self.recvneighbours.append([rank, [(local_y1 + diff_y1, local_x1 + diff_x1), (local_y2 - diff_y2, local_x2 - diff_x2)], 3])
			###########################################################################
			curr_y1 = y1
			curr_x1 = x2
			curr_y2 = y2
			curr_x2 = (x2 + overlap_x)

			if curr_y1 >= N_global_mod[0]:
				curr_y1 = curr_y1 % N_global_mod[0] 
				curr_y2 = curr_y2 % N_global_mod[0]

			if curr_x1 >= N_global_mod[1]:
				curr_x1 = curr_x1 % N_global_mod[1] 
				curr_x2 = curr_x2 % N_global_mod[1]

			if curr_y1 < 0:
				curr_y1 = curr_y1 % N_global_mod[0] 
				curr_y2 =           N_global_mod[0]

			if curr_x1 < 0:
				curr_x1 = curr_x1 % N_global_mod[1] 
				curr_x2 =           N_global_mod[1]

			res = self.detect_intersection(neigh_dc_mod, [(curr_y1, curr_x1),(curr_y2, curr_x2)])
			if res != None:
				diff_y1 = res[0][0] - curr_y1
				diff_x1 = res[0][1] - curr_x1
				diff_y2 = curr_y2 - res[1][0] 
				diff_x2 = curr_x2 - res[1][1] 

				local_y1 = overlap_y
				local_x1 = self.N_i_o[1] - overlap_x
				local_y2 = self.N_i_o[0] - overlap_y
				local_x2 = self.N_i_o[1]

				self.recvneighbours.append([rank, [(local_y1 + diff_y1, local_x1 + diff_x1), (local_y2 - diff_y2, local_x2 - diff_x2)], 4])
			###########################################################################
			curr_y1 = y2 
			curr_x1 = (x1 - overlap_x) 
			curr_y2 = (y2 + overlap_y) 
			curr_x2 = x1 

			if curr_y1 >= N_global_mod[0]:
				curr_y1 = curr_y1 % N_global_mod[0] 
				curr_y2 = curr_y2 % N_global_mod[0]

			if curr_x1 >= N_global_mod[1]:
				curr_x1 = curr_x1 % N_global_mod[1] 
				curr_x2 = curr_x2 % N_global_mod[1]

			if curr_y1 < 0:
				curr_y1 = curr_y1 % N_global_mod[0] 
				curr_y2 =           N_global_mod[0]

			if curr_x1 < 0:
				curr_x1 = curr_x1 % N_global_mod[1] 
				curr_x2 =           N_global_mod[1]

			res = self.detect_intersection(neigh_dc_mod, [(curr_y1, curr_x1),(curr_y2, curr_x2)])
			if res != None:
				diff_y1 = res[0][0] - curr_y1
				diff_x1 = res[0][1] - curr_x1
				diff_y2 = curr_y2 - res[1][0] 
				diff_x2 = curr_x2 - res[1][1] 

				local_y1 = self.N_i_o[0] - overlap_y
				local_x1 = 0
				local_y2 = self.N_i_o[0]
				local_x2 = overlap_x

				self.recvneighbours.append([rank, [(local_y1 + diff_y1, local_x1 + diff_x1), (local_y2 - diff_y2, local_x2 - diff_x2)], 5])
			###########################################################################
			curr_y1 = y2 
			curr_x1 = x1 
			curr_y2 = (y2 + overlap_y)
			curr_x2 = x2

			if curr_y1 >= N_global_mod[0]:
				curr_y1 = curr_y1 % N_global_mod[0] 
				curr_y2 = curr_y2 % N_global_mod[0]

			if curr_x1 >= N_global_mod[1]:
				curr_x1 = curr_x1 % N_global_mod[1] 
				curr_x2 = curr_x2 % N_global_mod[1]

			if curr_y1 < 0:
				curr_y1 = curr_y1 % N_global_mod[0] 
				curr_y2 =           N_global_mod[0]

			if curr_x1 < 0:
				curr_x1 = curr_x1 % N_global_mod[1] 
				curr_x2 =           N_global_mod[1]

			res = self.detect_intersection(neigh_dc_mod, [(curr_y1, curr_x1),(curr_y2, curr_x2)])
			if res != None:
				diff_y1 = res[0][0] - curr_y1
				diff_x1 = res[0][1] - curr_x1
				diff_y2 = curr_y2 - res[1][0] 
				diff_x2 = curr_x2 - res[1][1] 

				local_y1 = self.N_i_o[0] - overlap_y
				local_x1 = overlap_x
				local_y2 = self.N_i_o[0]
				local_x2 = self.N_i_o[1] - overlap_x

				self.recvneighbours.append([rank, [(local_y1 + diff_y1, local_x1 + diff_x1), (local_y2 - diff_y2, local_x2 - diff_x2)], 6])
			###########################################################################
			curr_y1 = y2 
			curr_x1 = x2 
			curr_y2 = (y2 + overlap_y) 
			curr_x2 = (x2 + overlap_x) 

			if curr_y1 >= N_global_mod[0]:
				curr_y1 = curr_y1 % N_global_mod[0] 
				curr_y2 = curr_y2 % N_global_mod[0]

			if curr_x1 >= N_global_mod[1]:
				curr_x1 = curr_x1 % N_global_mod[1] 
				curr_x2 = curr_x2 % N_global_mod[1]

			if curr_y1 < 0:
				curr_y1 = curr_y1 % N_global_mod[0] 
				curr_y2 =           N_global_mod[0]

			if curr_x1 < 0:
				curr_x1 = curr_x1 % N_global_mod[1] 
				curr_x2 =           N_global_mod[1]

			res = self.detect_intersection(neigh_dc_mod, [(curr_y1, curr_x1),(curr_y2, curr_x2)])
			if res != None:
				diff_y1 = res[0][0] - curr_y1
				diff_x1 = res[0][1] - curr_x1
				diff_y2 = curr_y2 - res[1][0] 
				diff_x2 = curr_x2 - res[1][1] 

				local_y1 = self.N_i_o[0] - overlap_y
				local_x1 = self.N_i_o[1] - overlap_x
				local_y2 = self.N_i_o[0]
				local_x2 = self.N_i_o[1]

				self.recvneighbours.append([rank, [(local_y1 + diff_y1, local_x1 + diff_x1), (local_y2 - diff_y2, local_x2 - diff_x2)], 7])

	def detect_intersection(self, recA, recB):
		x1 = max(recA[0][1], recB[0][1])
		y1 = max(recA[0][0], recB[0][0])
		x2 = min(recA[1][1], recB[1][1])
		y2 = min(recA[1][0], recB[1][0])
		if x1 < x2 and y1 < y2:
			return [(y1,x1), (y2,x2)]
		else:
			return None

	def get_next_velocity_index(self, direction):
		self.velocity_send_buffers_index[direction] = (self.velocity_send_buffers_index[direction] + 1) % self.velocity_send_buffers_num[direction]
		return self.velocity_send_buffers_index[direction]

	def get_next_density_index(self, direction):
		self.density_send_buffers_index[direction] = (self.density_send_buffers_index[direction] + 1) % self.density_send_buffers_num[direction]
		return self.density_send_buffers_index[direction]

	def resample_for_overlaps(self, result, key, y, correction = -1):
		if not key in result.keys():
			value = self.unique_resample_coeffs[key]
			new_size = (int(self.N_i_o[0] * value[0]), int(self.N_i_o[1] * value[1]))
			tmp     = y*self.bell
			if self.interpolation == 0:
				if correction == 0:
					result[key] = intr.interpfftn   (tmp, new_size, self.unique_spectral_uy_correction[key])
				if correction == 1:
					result[key] = intr.interpfftn   (tmp, new_size, self.unique_spectral_ux_correction[key])
				if correction == -1:
					result[key] = intr.interpfftn   (tmp, new_size)

			if self.interpolation > 0 and self.interpolation < 3:
				if correction == 0:
					result[key] = intr.interpregular_entire(tmp, new_size, self.method, (self.unique_uy_correction[key],0))
				if correction == 1:
					result[key] = intr.interpregular_entire(tmp, new_size, self.method, (0, self.unique_ux_correction[key]))
				if correction == -1:
					result[key] = intr.interpregular_entire(tmp, new_size, self.method)
			
			if self.interpolation > 2 and self.interpolation < 7:
				if correction == 0:
					result[key] = intr.interpspline_entire (tmp, new_size, self.order, (self.unique_uy_correction[key],0))
				if correction == 1:
					result[key] = intr.interpspline_entire (tmp, new_size, self.order, (0, self.unique_ux_correction[key]))
				if correction == -1:
					result[key] = intr.interpspline_entire (tmp, new_size, self.order)
				


	def insert_overlap(self, y, overlap_data, corners):
		y[corners[0][0]:corners[1][0], corners[0][1]:corners[1][1]] = np.copy(overlap_data)


	def get_overlap(self, y, corners):
		return np.copy(y[corners[0][0]:corners[1][0], corners[0][1]:corners[1][1]])

	def get_overlap_special(self, y, corners, res_coeff, correction = -1):
		ralative_res_coeff = self.unique_resample_coeffs[res_coeff]
		if self.interpolation > 6 and self.interpolation < 9:
			if correction == 0:
				return np.copy (intr.interpregular(y, corners, ralative_res_coeff, self.method, (self.unique_uy_correction[res_coeff],0)))
			if correction == 1:
				return np.copy (intr.interpregular(y, corners, ralative_res_coeff, self.method, (0, self.unique_ux_correction[res_coeff])))
			if correction == -1:
				return np.copy (intr.interpregular(y, corners, ralative_res_coeff, self.method))
		
		if self.interpolation > 8 and self.interpolation < 13:
			if correction == 0:
				return np.copy (intr.interpspline  (y, corners, ralative_res_coeff, self.order, (self.unique_uy_correction[res_coeff],0)))
			if correction == 1:
				return np.copy (intr.interpspline  (y, corners, ralative_res_coeff, self.order, (0, self.unique_ux_correction[res_coeff])))
			if correction == -1:
				return np.copy (intr.interpspline  (y, corners, ralative_res_coeff, self.order))

	def scatter_overlaps_stress(self, y, iteration, printing=False):

		# self.prepare_overlaps_stress(y, timestep)
		resampled_data_map0 = {}
		resampled_data_map1 = {}
		resampled_data_map2 = {}
		resampled_data_map3 = {}
		resampled_data_map4 = {}
		resampled_data_map5 = {}

		work = False
		for i, entry in enumerate(self.sendneighbours):
			neighbour_dt = self.dir_neighbour_dt[i]
			for neighbour_timestamp, neighbour_iteration in self.neigbour_b_step_index_todo[neighbour_dt][iteration]:
				if (neighbour_iteration % self.dir_neighbour_comm_freq[i]) != 0:
					continue
				work = True
		if not work:
			return

		for i, entry in enumerate(self.sendneighbours):
			tag_dir = entry[2]
			if self.single_dt:
				if self.single_res:
					resampled_data_map0[(1.0,1.0)] = y[0]
					resampled_data_map1[(1.0,1.0)] = y[1]
					resampled_data_map2[(1.0,1.0)] = y[2]
				else:
					self.resample_for_overlaps(resampled_data_map0, self.dir_neighbour_res_coeffs[i], y[0])
					self.resample_for_overlaps(resampled_data_map1, self.dir_neighbour_res_coeffs[i], y[1])
					self.resample_for_overlaps(resampled_data_map2, self.dir_neighbour_res_coeffs[i], y[2])

				neighbour_dt = self.dir_neighbour_dt[i]
				for neighbour_timestamp, neighbour_iteration in self.neigbour_b_step_index_todo[neighbour_dt][iteration]:
					if (neighbour_iteration % self.dir_neighbour_comm_freq[i]) != 0:
						continue

					index = self.get_next_density_index(i)

					tag_time_stamp = neighbour_iteration<<VAR_BITS

					if self.interpolation < 7:
						sxx = self.get_overlap(resampled_data_map0[self.dir_neighbour_res_coeffs[i]], entry[1])
						syy = self.get_overlap(resampled_data_map1[self.dir_neighbour_res_coeffs[i]], entry[1])
						sxy = self.get_overlap(resampled_data_map2[self.dir_neighbour_res_coeffs[i]], entry[1])
					else:
						sxx = self.get_overlap_special(y[0], entry[1], self.dir_neighbour_res_coeffs[i])
						syy = self.get_overlap_special(y[1], entry[1], self.dir_neighbour_res_coeffs[i])
						sxy = self.get_overlap_special(y[2], entry[1], self.dir_neighbour_res_coeffs[i])


					if self.dir_send_neighbour_model[i] == ELASTIC_MODEL:
						self.sxx_overlap_send_buf_rq[i][index].Wait()
						self.sxx_overlap_send_buf[i][index] = np.copy(sxx)
						self.sxx_overlap_send_buf_rq[i][index] = self.comm.Isend(self.sxx_overlap_send_buf[i][index], dest=entry[0], tag=tag_time_stamp+SXX+tag_dir)

						self.syy_overlap_send_buf_rq[i][index].Wait()
						self.syy_overlap_send_buf[i][index] = np.copy(syy)
						self.syy_overlap_send_buf_rq[i][index] = self.comm.Isend(self.syy_overlap_send_buf[i][index], dest=entry[0], tag=tag_time_stamp+SYY+tag_dir)
						
						self.sxy_overlap_send_buf_rq[i][index].Wait()
						self.sxy_overlap_send_buf[i][index] = np.copy(sxy)
						self.sxy_overlap_send_buf_rq[i][index] = self.comm.Isend(self.sxy_overlap_send_buf[i][index], dest=entry[0], tag=tag_time_stamp+SXY+tag_dir)
					else:
						self.rho_overlap_send_buf_rq[i][index].Wait()
						if not np.isscalar(self.medium_propperties["sound_speed_compression"]):
							self.rho_overlap_send_buf[i][index] = (-(sxx + syy)/2.0) / self.get_overlap((self.medium_propperties["sound_speed_compression"]**2), entry[1])
						else:	
							self.rho_overlap_send_buf[i][index] = (-(sxx + syy)/2.0) / (self.medium_propperties["sound_speed_compression"]**2)
						self.rho_overlap_send_buf_rq[i][index] = self.comm.Isend(self.rho_overlap_send_buf[i][index], dest=entry[0], tag=tag_time_stamp+P+tag_dir)
			else:
				if self.single_res:
					resampled_data_map0[(1.0,1.0)] = y[3]
					resampled_data_map1[(1.0,1.0)] = y[4]
					resampled_data_map2[(1.0,1.0)] = y[5]
					resampled_data_map3[(1.0,1.0)] = y[6]
					resampled_data_map4[(1.0,1.0)] = y[7]
					resampled_data_map5[(1.0,1.0)] = y[8]
				else:
					self.resample_for_overlaps(resampled_data_map0, self.dir_neighbour_res_coeffs[i], y[3])
					self.resample_for_overlaps(resampled_data_map1, self.dir_neighbour_res_coeffs[i], y[4])
					self.resample_for_overlaps(resampled_data_map2, self.dir_neighbour_res_coeffs[i], y[5])
					self.resample_for_overlaps(resampled_data_map3, self.dir_neighbour_res_coeffs[i], y[6])
					self.resample_for_overlaps(resampled_data_map4, self.dir_neighbour_res_coeffs[i], y[7])
					self.resample_for_overlaps(resampled_data_map5, self.dir_neighbour_res_coeffs[i], y[8])

				neighbour_dt = self.dir_neighbour_dt[i]
				for neighbour_timestamp, neighbour_iteration in self.neigbour_b_step_index_todo[neighbour_dt][iteration]:
					if (neighbour_iteration % self.dir_neighbour_comm_freq[i]) != 0:
						continue
					
					index = self.get_next_density_index(i)
					
					mod_dt = neighbour_timestamp - ((iteration-1)*self.dt)

					if self.interpolation < 7:
						sxx_old = self.get_overlap(resampled_data_map0[self.dir_neighbour_res_coeffs[i]], entry[1])
						syy_old = self.get_overlap(resampled_data_map1[self.dir_neighbour_res_coeffs[i]], entry[1])
						sxy_old = self.get_overlap(resampled_data_map2[self.dir_neighbour_res_coeffs[i]], entry[1])

						duxx    = self.get_overlap(resampled_data_map3[self.dir_neighbour_res_coeffs[i]], entry[1])
						duyy    = self.get_overlap(resampled_data_map4[self.dir_neighbour_res_coeffs[i]], entry[1])
						duxy    = self.get_overlap(resampled_data_map5[self.dir_neighbour_res_coeffs[i]], entry[1])
					else:
						sxx_old = self.get_overlap_special(y[3], entry[1], self.dir_neighbour_res_coeffs[i])
						syy_old = self.get_overlap_special(y[4], entry[1], self.dir_neighbour_res_coeffs[i])
						sxy_old = self.get_overlap_special(y[5], entry[1], self.dir_neighbour_res_coeffs[i])

						duxx    = self.get_overlap_special(y[6], entry[1], self.dir_neighbour_res_coeffs[i])
						duyy    = self.get_overlap_special(y[7], entry[1], self.dir_neighbour_res_coeffs[i])
						duxy    = self.get_overlap_special(y[8], entry[1], self.dir_neighbour_res_coeffs[i])
					
					tag_time_stamp = neighbour_iteration<<VAR_BITS
					
					if self.dir_send_neighbour_model[i] == ELASTIC_MODEL:
						self.sxx_overlap_send_buf_rq[i][index].Wait()
						self.sxx_overlap_send_buf[i][index] = np.copy(sxx_old + mod_dt * duxx)
						self.sxx_overlap_send_buf_rq[i][index] = self.comm.Isend(self.sxx_overlap_send_buf[i][index], dest=entry[0], tag=tag_time_stamp+SXX+tag_dir)

						self.syy_overlap_send_buf_rq[i][index].Wait()
						self.syy_overlap_send_buf[i][index] = np.copy(syy_old + mod_dt * duyy)
						self.syy_overlap_send_buf_rq[i][index] = self.comm.Isend(self.syy_overlap_send_buf[i][index], dest=entry[0], tag=tag_time_stamp+SYY+tag_dir)

						self.sxy_overlap_send_buf_rq[i][index].Wait()
						self.sxy_overlap_send_buf[i][index] = np.copy(sxy_old + mod_dt * duxy)
						self.sxy_overlap_send_buf_rq[i][index] = self.comm.Isend(self.sxy_overlap_send_buf[i][index], dest=entry[0], tag=tag_time_stamp+SXY+tag_dir)
					else:
						self.rho_overlap_send_buf_rq[i][index].Wait()
						p = -((sxx_old + mod_dt * duxx) + (syy_old + mod_dt * duyy))/ (2.0)
						if not np.isscalar(self.medium_propperties["sound_speed_compression"]):
							self.rho_overlap_send_buf[i][index] = p / self.get_overlap((self.medium_propperties["sound_speed_compression"]**2), entry[1])
						else:
							self.rho_overlap_send_buf[i][index] = p / (self.medium_propperties["sound_speed_compression"]**2)
						self.rho_overlap_send_buf_rq[i][index] = self.comm.Isend(self.rho_overlap_send_buf[i][index], dest=entry[0], tag=tag_time_stamp+P+tag_dir)

	def gather_overlaps_stress(self, y, iteration, printing=False):
		if (iteration % self.comm_freq) != 0:
			return

		loc_timestep = iteration<<VAR_BITS
		for i, entry in enumerate(self.recvneighbours):
			tag_dir = entry[2]
			if self.dir_recv_neighbour_model[i] == ELASTIC_MODEL:
				self.comm.Recv(self.sxx_overlap_recv_buf[i], source=entry[0], tag=loc_timestep+SXX+tag_dir)
				self.comm.Recv(self.syy_overlap_recv_buf[i], source=entry[0], tag=loc_timestep+SYY+tag_dir)
				self.comm.Recv(self.sxy_overlap_recv_buf[i], source=entry[0], tag=loc_timestep+SXY+tag_dir)


				self.insert_overlap(y[0], self.sxx_overlap_recv_buf[i], entry[1])
				self.insert_overlap(y[1], self.syy_overlap_recv_buf[i], entry[1])
				self.insert_overlap(y[2], self.sxy_overlap_recv_buf[i], entry[1])
			else:
				self.comm.Recv(self.rho_overlap_recv_buf[i], source=entry[0], tag=loc_timestep+P+tag_dir)
				if not np.isscalar(self.medium_propperties["sound_speed_compression"]):
					p = self.rho_overlap_recv_buf[i] * self.get_overlap((self.medium_propperties["sound_speed_compression"]**2), entry[1])
				else:
					p = self.rho_overlap_recv_buf[i] * (self.medium_propperties["sound_speed_compression"]**2)
				self.insert_overlap(y[0], -p, entry[1])
				self.insert_overlap(y[1], -p, entry[1])
	

	def scatter_overlaps_density(self, y, iteration, printing=False):

		# self.prepare_overlaps_density(y, timestep)
		resampled_data_map0 = {}
		resampled_data_map1 = {}

		work = False
		for i, entry in enumerate(self.sendneighbours):
			neighbour_dt = self.dir_neighbour_dt[i]
			for neighbour_timestamp, neighbour_iteration in self.neigbour_b_step_index_todo[neighbour_dt][iteration]:
				if (neighbour_iteration % self.dir_neighbour_comm_freq[i]) != 0:
					continue
				work = True
		if not work:
			return


		for i, entry in enumerate(self.sendneighbours):
			tag_dir = entry[2]
			if self.single_dt:
				if self.single_res:
					resampled_data_map0[(1.0,1.0)] = y[0]
				else:
					self.resample_for_overlaps(resampled_data_map0, self.dir_neighbour_res_coeffs[i], y[0])
				

				neighbour_dt = self.dir_neighbour_dt[i]
				for neighbour_timestamp, neighbour_iteration in self.neigbour_b_step_index_todo[neighbour_dt][iteration]:
					if (neighbour_iteration % self.dir_neighbour_comm_freq[i]) != 0:
						continue
					
					index = self.get_next_density_index(i)

					tag_time_stamp = neighbour_iteration<<VAR_BITS
					
					if self.interpolation < 7:
						p = self.get_overlap(resampled_data_map0[self.dir_neighbour_res_coeffs[i]], entry[1])
					else:
						p = self.get_overlap_special(y[0], entry[1], self.dir_neighbour_res_coeffs[i])

					self.rho_overlap_send_buf_rq[i][index].Wait()
					self.rho_overlap_send_buf[i][index] = np.copy(p)
					self.rho_overlap_send_buf_rq[i][index] = self.comm.Isend(self.rho_overlap_send_buf[i][index], dest=entry[0], tag=tag_time_stamp+P+tag_dir)

			else:
				if self.single_res:
					resampled_data_map0[(1.0,1.0)] = y[1]
					resampled_data_map1[(1.0,1.0)] = y[2]
				else:
					self.resample_for_overlaps(resampled_data_map0, self.dir_neighbour_res_coeffs[i], y[1])
					self.resample_for_overlaps(resampled_data_map1, self.dir_neighbour_res_coeffs[i], y[2])

				neighbour_dt = self.dir_neighbour_dt[i]
				for neighbour_timestamp, neighbour_iteration in self.neigbour_b_step_index_todo[neighbour_dt][iteration]:
					if (neighbour_iteration % self.dir_neighbour_comm_freq[i]) != 0:
						continue

					tag_time_stamp = neighbour_iteration<<VAR_BITS
					
					index = self.get_next_density_index(i)

					mod_dt = neighbour_timestamp - ((iteration-1)*self.dt)
					
					if self.interpolation < 7:
						rho_old = self.get_overlap(resampled_data_map0[self.dir_neighbour_res_coeffs[i]], entry[1])
						du      = self.get_overlap(resampled_data_map1[self.dir_neighbour_res_coeffs[i]], entry[1])
					else:
						rho_old = self.get_overlap_special(y[1], entry[1], self.dir_neighbour_res_coeffs[i] )
						du      = self.get_overlap_special(y[2], entry[1], self.dir_neighbour_res_coeffs[i] )

					self.rho_overlap_send_buf_rq[i][index].Wait()
					self.rho_overlap_send_buf[i][index] = np.copy(rho_old - mod_dt * du)
					self.rho_overlap_send_buf_rq[i][index] = self.comm.Isend(self.rho_overlap_send_buf[i][index], dest=entry[0], tag=tag_time_stamp+P+tag_dir)


	def gather_overlaps_density(self, y, iteration, printing=False):
		if (iteration % self.comm_freq) != 0:
			return
		loc_timestep = iteration<<VAR_BITS
		for i, entry in enumerate(self.recvneighbours):
			tag_dir = entry[2]
			self.comm.Recv(self.rho_overlap_recv_buf[i], source=entry[0], tag=loc_timestep+P+tag_dir)		
			self.insert_overlap(y[0], self.rho_overlap_recv_buf[i], entry[1])

	def scatter_overlaps_velocity(self, y, iteration, printing=False):
		
		resampled_data_map0 = {}
		resampled_data_map1 = {}
		resampled_data_map2 = {}
		resampled_data_map3 = {}

		work = False
		for i, entry in enumerate(self.sendneighbours):
			neighbour_dt = self.dir_neighbour_dt[i]
			for neighbour_timestamp, neighbour_iteration in self.neigbour_a_step_index_todo[neighbour_dt][iteration]:
				if (neighbour_iteration % self.dir_neighbour_comm_freq[i]) != 0:
					continue
				work = True
		if not work:
			return

		for i, entry in enumerate(self.sendneighbours):
			tag_dir = entry[2]
			
			if self.single_dt:
				if self.single_res:
					resampled_data_map0[(1.0,1.0)] = y[0]
					resampled_data_map1[(1.0,1.0)] = y[1]
				else:
					self.resample_for_overlaps(resampled_data_map0, self.dir_neighbour_res_coeffs[i], y[0], 1)
					self.resample_for_overlaps(resampled_data_map1, self.dir_neighbour_res_coeffs[i], y[1], 0)

				neighbour_dt = self.dir_neighbour_dt[i]
				for neighbour_timestamp, neighbour_iteration in self.neigbour_a_step_index_todo[neighbour_dt][iteration]:
					if (neighbour_iteration % self.dir_neighbour_comm_freq[i]) != 0:
						continue

					index = self.get_next_velocity_index(i)
					
					tag_time_stamp = neighbour_iteration<<VAR_BITS

					if self.interpolation < 7:
						ux = self.get_overlap(resampled_data_map0[self.dir_neighbour_res_coeffs[i]], entry[1])
						uy = self.get_overlap(resampled_data_map1[self.dir_neighbour_res_coeffs[i]], entry[1])
					else:
						ux = self.get_overlap_special(y[0], entry[1], self.dir_neighbour_res_coeffs[i], 1)
						uy = self.get_overlap_special(y[1], entry[1], self.dir_neighbour_res_coeffs[i], 0)

					
					self.ux_overlap_send_buf_rq[i][index].Wait()
					self.ux_overlap_send_buf[i][index] = np.copy(ux)
					self.ux_overlap_send_buf_rq[i][index] = self.comm.Isend(self.ux_overlap_send_buf[i][index], dest=entry[0], tag=tag_time_stamp+UX+tag_dir)

					self.uy_overlap_send_buf_rq[i][index].Wait()
					self.uy_overlap_send_buf[i][index] = np.copy(uy)
					self.uy_overlap_send_buf_rq[i][index] = self.comm.Isend(self.uy_overlap_send_buf[i][index], dest=entry[0], tag=tag_time_stamp+UY+tag_dir)
			else:
				if self.single_res:
					resampled_data_map0[(1.0,1.0)] = y[2]
					resampled_data_map1[(1.0,1.0)] = y[3]
					resampled_data_map2[(1.0,1.0)] = y[4]
					resampled_data_map3[(1.0,1.0)] = y[5]
				else:
					self.resample_for_overlaps(resampled_data_map0, self.dir_neighbour_res_coeffs[i], y[2], 1)
					self.resample_for_overlaps(resampled_data_map1, self.dir_neighbour_res_coeffs[i], y[3], 0)
					self.resample_for_overlaps(resampled_data_map2, self.dir_neighbour_res_coeffs[i], y[4], 1)
					self.resample_for_overlaps(resampled_data_map3, self.dir_neighbour_res_coeffs[i], y[5], 0)

				neighbour_dt = self.dir_neighbour_dt[i]

				for neighbour_timestamp, neighbour_iteration in self.neigbour_a_step_index_todo[neighbour_dt][iteration]:
					if (neighbour_iteration % self.dir_neighbour_comm_freq[i]) != 0:
						continue

					index = self.get_next_velocity_index(i)

					tag_time_stamp = neighbour_iteration<<VAR_BITS

					mod_dt = neighbour_timestamp - (iteration*self.dt - self.dt/2)

					if self.interpolation < 7:
						ux_old = self.get_overlap(resampled_data_map0[self.dir_neighbour_res_coeffs[i]], entry[1])
						uy_old = self.get_overlap(resampled_data_map1[self.dir_neighbour_res_coeffs[i]], entry[1])

						dsx    = self.get_overlap(resampled_data_map2[self.dir_neighbour_res_coeffs[i]], entry[1])
						dsy    = self.get_overlap(resampled_data_map3[self.dir_neighbour_res_coeffs[i]], entry[1])
					else:
						ux_old = self.get_overlap_special(y[2], entry[1], self.dir_neighbour_res_coeffs[i], 1)
						uy_old = self.get_overlap_special(y[3], entry[1], self.dir_neighbour_res_coeffs[i], 0)

						dsx    = self.get_overlap_special(y[4], entry[1], self.dir_neighbour_res_coeffs[i], 1)
						dsy    = self.get_overlap_special(y[5], entry[1], self.dir_neighbour_res_coeffs[i], 0)
					
					self.ux_overlap_send_buf_rq[i][index].Wait()
					self.ux_overlap_send_buf[i][index] = np.copy(ux_old - mod_dt * dsx)
					self.ux_overlap_send_buf_rq[i][index] = self.comm.Isend(self.ux_overlap_send_buf[i][index], dest=entry[0], tag=tag_time_stamp+UX+tag_dir)

					self.uy_overlap_send_buf_rq[i][index].Wait()
					self.uy_overlap_send_buf[i][index] = np.copy(uy_old - mod_dt * dsy)
					self.uy_overlap_send_buf_rq[i][index] = self.comm.Isend(self.uy_overlap_send_buf[i][index], dest=entry[0], tag=tag_time_stamp+UY+tag_dir)

		
	def gather_overlaps_velocity(self, y, iteration, printing=False):
		if (iteration % self.comm_freq) != 0:
			return
		loc_timestep = iteration<<VAR_BITS
		for i, entry in enumerate(self.recvneighbours):
			tag_dir = entry[2]
			self.comm.Recv(self.ux_overlap_recv_buf[i], source=entry[0], tag=loc_timestep+UX+tag_dir)
			self.comm.Recv(self.uy_overlap_recv_buf[i], source=entry[0], tag=loc_timestep+UY+tag_dir)


			self.insert_overlap(y[0], self.ux_overlap_recv_buf[i], entry[1])
			self.insert_overlap(y[1], self.uy_overlap_recv_buf[i], entry[1])
	def gather_global_result(self, y):
		tmp = intr.interpfftn(y*self.bell, (self.N_base[0]+2*self.base_overlap[0], self.N_base[1]+2*self.base_overlap[1]))
		self.sendbuf = np.copy(tmp[self.base_overlap[0]:-self.base_overlap[0], self.base_overlap[1]:-self.base_overlap[1]].flatten())
		self.comm.Gatherv(sendbuf=self.sendbuf, recvbuf=(self.recvbuf, self.sendcounts), root=0)

	def get_global_result(self):
		if self.rank == 0:	
			ret = np.empty(self.N_global)
			for i in range(0, self.num_domains):
				y = np.reshape(self.recvbuf[self.sendcounts_scan[i] : self.sendcounts_scan[i+1]], [self.d_cs[i][1][0] - self.d_cs[i][0][0], self.d_cs[i][1][1] - self.d_cs[i][0][1]])
				ret[self.d_cs[i][0][0] : self.d_cs[i][1][0], self.d_cs[i][0][1]: self.d_cs[i][1][1]] = y
			return ret
		else:
			return None

	def get_inner_part(self, y):
		return np.copy(y[self.overlap[0]:-self.overlap[0], self.overlap[1]:-self.overlap[1]])

	def get_global_slice(self, y):

		if not self.on_base_res:
			y = intr.interpfftn(y, (int(self.N_global[0] * self.res_coeff[0]), int(self.N_global[1] * self.res_coeff[1])))

		ret = np.zeros((self.N[0] + 2*self.overlap[0], self.N[1] + 2*self.overlap[1]))

		start_y = int(self.d_c[0][0]*self.res_coeff[0])
		end_y   = int(self.d_c[1][0]*self.res_coeff[0])

		start_x = int(self.d_c[0][1]*self.res_coeff[1])
		end_x   = int(self.d_c[1][1]*self.res_coeff[1])

		
		tmp = np.copy(y[start_y : end_y, start_x : end_x])

		ret[self.overlap[0]:self.N[0] + self.overlap[0], self.overlap[1]:self.N[1] + self.overlap[1]] = tmp


		return ret

	def get_bell(self, bell_size, taper_size, param=2):

		# x = np.arange(-1, 1, 2/(taper_size))
		x = np.linspace(-1, 1, taper_size, endpoint=False)
		H = 0.5*(1 + special.erf(param * x / np.sqrt( 1 - x**2 )) )
		bell = np.ones(bell_size);
		bell[0:taper_size] = H;
		bell[-taper_size : bell_size] = np.flip(H,0);
		return bell

	def get_spectral_shift(self, shift_amout, direction):
		if direction == 0:
			size  = self.N[0] + 2*self.overlap[0]
			shape = (size, 1)
			ds    = self.ds[0]
		
		if direction == 1:
			size  = self.N[1] + 2*self.overlap[1]
			shape = (1, size)
			ds    = self.ds[1]

		n_vec = np.linspace(-0.5, 0.5, size, endpoint=False)
		ks_vec = (2*np.pi/ds)*n_vec.reshape(shape)
		
		return t.ifftshift(np.exp( 1j*ks_vec*shift_amout))

	def get_spectral_deriv(self,  direction):
		if direction == 0:
			size  = self.N[0] + 2*self.overlap[0]
			shape = (size, 1)
			ds    = self.ds[0]
		
		if direction == 1:
			size  = self.N[1] + 2*self.overlap[1]
			shape = (1, size)
			ds    = self.ds[1]

		n_vec = np.linspace(-0.5, 0.5, size, endpoint=False)
		ks_vec = (2*np.pi/ds)*n_vec.reshape(shape)
		
		return t.ifftshift(1j*ks_vec)
	#TODO otestovat 

	def get_k(self, c_ref):
		
		size_y  = self.N[0] + 2*self.overlap[0]
		shape_y = (size_y, 1)
		ds_y    = self.ds[0]

		y_vec  = np.linspace(-0.5, 0.5, size_y, endpoint=False)
		ky_vec = (2*np.pi/ds_y)*y_vec.reshape(shape_y)

		size_x  = self.N[1] + 2*self.overlap[1]
		shape_x = (1, size_x)
		ds_x    = self.ds[1]

		x_vec  = np.linspace(-0.5, 0.5, size_x, endpoint=False)
		kx_vec = (2*np.pi/ds_x)*x_vec.reshape(shape_x)

		return np.sqrt(np.square(np.tile(ky_vec, shape_x)) + np.square(np.tile(kx_vec, shape_y)))

	def get_kappa(self, c_ref):
		
		size_y  = self.N[0] + 2*self.overlap[0]
		shape_y = (size_y, 1)
		ds_y    = self.ds[0]

		y_vec  = np.linspace(-0.5, 0.5, size_y, endpoint=False)
		ky_vec = (2*np.pi/ds_y)*y_vec.reshape(shape_y)

		size_x  = self.N[1] + 2*self.overlap[1]
		shape_x = (1, size_x)
		ds_x    = self.ds[1]

		x_vec  = np.linspace(-0.5, 0.5, size_x, endpoint=False)
		kx_vec = (2*np.pi/ds_x)*x_vec.reshape(shape_x)

		k = np.sqrt(np.square(np.tile(ky_vec, shape_x)) + np.square(np.tile(kx_vec, shape_y)))

		x = c_ref * k * self.dt / 2
		zero_vals = (x == 0)
		return t.ifftshift(np.sin(x + np.pi * zero_vals) / (x + zero_vals) + zero_vals)

	def prepare_medium_propperties(self, variables):
		self.medium_propperties = {}
		# self.sound_speed_compression_send = []
		# self.sound_speed_compression_recv = []
		for key, value in variables.items():
			if key.find("inv") >= 0:
				is_inverted = True
			else:
				is_inverted = False

			if key.find("sgxy") >= 0:
				is_sgxy = True
			else:
				is_sgxy = False

			if key.find("sgx") >= 0:
				is_sgx = True
			else:
				is_sgx = False

			if key.find("sgy") >= 0:
				is_sgy = True
			else:
				is_sgy = False

			if key.find("geom") >= 0:
				geom = 1
			else:
				geom = 0

			if is_sgxy:
				inter_value = intr.interplinear(value, self.X_SGX, self.Y_SGY, geom)
			elif is_sgx:
				inter_value = intr.interplinear(value, self.X_SGX, self.Y    , geom)
			elif is_sgy:
				inter_value = intr.interplinear(value, self.X    , self.Y_SGY, geom)
			else:
				inter_value = intr.interplinear(value, self.X    , self.Y    , geom)

			if is_inverted:
				inter_value = 1.0 / inter_value

			if not np.isscalar(inter_value):
				self.medium_propperties[key] = inter_value.reshape(self.N_i_o)
			else:
				self.medium_propperties[key] = inter_value

			# if key.find("sound_speed_compression") >= 0:
			# 	if not np.isscalar(inter_value):
			# 		self.medium_propperties[key] = inter_value.reshape(self.N_i_o)
			# 	else:
			# 		self.medium_propperties[key] = inter_value
		return self.medium_propperties
	
	def prepare_local_sources(self, source_indices, source_values):
		x = []
		y = []
		values = []
		for i in range(len(source_indices[0])):
			if source_indices[0][i] >= self.d_c[0][0] and \
			   source_indices[0][i] <= self.d_c[1][0] and \
			   source_indices[1][i] >= self.d_c[0][1] and \
			   source_indices[1][i] <= self.d_c[1][1]:
				y.append(source_indices[0][i] - self.d_c[0][0])
				x.append(source_indices[1][i] - self.d_c[0][1])
				values.append(source_values[i])

		return ((np.array(x, dtype=np.int32),np.array(y, dtype=np.int32)), np.array(values, dtype=np.int32))
		
