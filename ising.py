import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import sys
from sys import argv
import configparser
from pathlib import Path
import argparse

config = None

def parse_command_line():

	parser = argparse.ArgumentParser(description ='Simulate Ising Model using Metropolis Algorithm')

	parser.add_argument('-f', '--file',  
						help ='path to config file', type = str)

	parser.add_argument('-e', '--experiment',  help ='Run a specific experiment from the config file', type=str)

	args = parser.parse_args()
	
	return args


def parse_config_file():

	args = parse_command_line()
	
	config = configparser.ConfigParser()

	file = None;
	if (args.file):
		print("Reading config file name from command line")
		file = Path(args.file)
		if not file.is_file():
			print(f"Config file {args.file} not found. Exiting.")
			sys.exit(1)
	else:
		file = Path("./config.ini")
		if file.is_file():
			print("Reading default 'config.ini' from current directory")
			file = "./config.ini"
		else:
			print("No config file found in current directory. Exiting.")
			sys.exit(1)


	config.read(file)

	section = args.experiment
	if (section == None):
		print(f"Reading config file for default section [ising]")
		c =	read_config_experiment(config, 'ising')
	elif (config.has_section(section)):
		print(f"Reading config file for section [{section}]")
		c =	read_config_experiment(config, section)
	else:
		print(f"Section {section} not found in config file. Exiting.")
		sys.exit(1)


	print(f'''Running the experiment with following configs:\n
		Seed={c['seed']}, Lattice Dimension={c['D']},
		Temperatures ranging from {c['initial_temp']} to {c['final_temp']} with step size of {c['temp_steps']}
		For Finite Size Scaling, Dimensions ranging from {c['intial_lattice']} to {c['final_lattice']} with step size of {c['lattice_step']}
	''')
	
	return c

def read_config_experiment(config, name):

	c = dict()
	c['seed'] = config.getint(name, 'random_number_seed')

	c['T_c'] = config.getfloat('constant', 'Tc')
	c['D'] = config.getint(name, 'D')
	c['intial_lattice'] = config.getint(name, 'intial_lattice')
	c['final_lattice'] = config.getint(name, 'final_lattice')
	c['lattice_step'] = config.getint(name, 'lattice_step')
	c['initial_temp']  = config.getfloat(name, 'initial_temp')
	c['final_temp'] = config.getfloat(name, 'final_temp')
	c['temp_steps'] = config.getint(name, 'temp_steps')

	return c

def initialize_spins(D):
	'''
	This function initializes the spins of the lattice with random values of 1 and -1

	Parameters:
	D (int): Size of the lattice
	seed (int): Seed for random number generator

	Returns:
	spins (np.array): The lattice of spins
	'''
	global config
	seed = config['seed']
	rng = np.random.default_rng(seed)
	return rng.choice([1,-1], size=(config['D'], config['D']))

def compute_energy(spins, J=1.0):
	'''
	This function computes the energy of the lattice using the Ising Model Hamiltonian
	
	Parameters:
	spins (np.array): The lattice of spins
	J (float): Coupling constant
	
	Returns:
	energy (float): The energy of the lattice
	'''

	top = np.roll(spins, 1, axis=0)
	bottom = np.roll(spins, -1, axis=0)
	left = np.roll(spins, 1, axis=1)
	right = np.roll(spins, -1, axis=1)
	neighbor_sum = top + bottom + left + right
	return -J * np.sum(spins * neighbor_sum) / 2

def metropolis_step(spins, T, J=1.0, kB=1.0):
	'''
	This function performs a single Metropolis step on the lattice of spins
	
	Parameters:
	spins (np.array): The lattice of spins
	T (float): Temperature
	J (float): Coupling constant
	kB (float): Boltzmann constant
	
	Returns:
	spins (np.array): The updated lattice of spins'''

	D = spins.shape[0]

	for _ in range(D * D):
		i, j = np.random.randint(D), np.random.randint(D)
		current_spin = spins[i, j]

		top = spins[(i - 1) % D, j]
		bottom = spins[(i + 1) % D, j]
		left = spins[i, (j - 1) % D]
		right = spins[i, (j + 1) % D]
		sum_neighbors = top + bottom + left + right

		delta_E = 2 * J * current_spin * sum_neighbors
		
		if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / (kB * T)):
				spins[i, j] *= -1

	return spins

def simulate(D, T, eq_steps=500, mc_steps=500, J=1.0, kB=1.0):
	'''
	This function simulates the Ising Model for a given lattice size and temperature
	
	Parameters:
	D (int): Size of the lattice
	T (float): Temperature of the system
	eq_steps (int): Number of equilibrium steps
	mc_steps (int): Number of Monte Carlo steps
	J (float): Coupling constant
	kB (float): Boltzmann constant
	
	Returns:
	avg_M (float): Average magnetization
	C (float): Heat capacity
	chi (float): Susceptibility
	'''
	global config
	spins = initialize_spins(D)

	energy_sum = energy_sq_sum = mag_sum = mag_sq_sum = 0.0
	
	for _ in tqdm(range(eq_steps), desc="Equilibrium Steps"):
		spins = metropolis_step(spins, T, J, kB)

	for _ in tqdm(range(mc_steps), desc="Monte Carlo Moves"):
			spins = metropolis_step(spins, T, J, kB)
			E = compute_energy(spins, J)
			M = np.mean(spins)
			energy_sum += E
			energy_sq_sum += E**2
			mag_sum += abs(M)
			mag_sq_sum += M**2
	
	avg_E = energy_sum / mc_steps
	avg_E2 = energy_sq_sum / mc_steps
	avg_M = mag_sum / mc_steps
	avg_M2 = mag_sq_sum / mc_steps
	C = (avg_E2 - avg_E**2) / (kB * T**2)
	chi = (avg_M2 - avg_M**2) / (kB * T)

	return avg_M, C, chi


# ====================== MAIN SIMULATION ======================
def main():
	'''
	This function is the entry point for the Ising Model simulation

	It calculates the properties of the Ising Model, performs Finite-Size Scaling of Heat Capacity at T = T_c,
	calculates the log-log values for Magnetization and Susceptibility, and plots the results

	Parameters:
	None

	Returns:
	None
	'''

	global config 
	config = parse_config_file()

	temperatures, magnetizations, heat_capacities, susceptibilities = calculate_properties()
	C_values, D_values, T_c = finite_size_scaling()
	log_D, params, fit_C, T_near_Tc, M_near_Tc, Chi_near_Tc = calculate_log_log(D_values, T_c, temperatures, magnetizations, susceptibilities, C_values)
	plot_all(T_c, log_D, C_values, params, fit_C, T_near_Tc, M_near_Tc, Chi_near_Tc, temperatures, magnetizations, heat_capacities, susceptibilities)

def calculate_properties():
	'''
	This function calculates the properties of the Ising Model for a given lattice size, such as magnetization, heat capacity, and susceptibility
	
	Parameters:
	D (int): Size of the lattice
	
	Returns:
	temperatures (np.array): Array of temperatures
	magnetizations (np.array): Array of magnetizations
	heat_capacities (np.array): Array of heat capacities
	susceptibilities (np.array): Array of susceptibilities'''

	global config
	D = config['D']
	temperatures = np.linspace(config['initial_temp'], config['final_temp'], config['temp_steps'])
	magnetizations, heat_capacities, susceptibilities = [], [], []

	print(f"\nIterating over {config['temp_steps']} Temperature values:\n")
	for T in tqdm(temperatures, desc="Temperature: "):
			M, C, chi = simulate(D, T)
			magnetizations.append(M)
			heat_capacities.append(C)
			susceptibilities.append(chi)

	return temperatures, magnetizations, heat_capacities, susceptibilities

#Finite-Size Scaling of Heat Capacity at T = T_c
def finite_size_scaling():
	'''
	This function performs Finite-Size Scaling of Heat Capacity at T = T_c
	
	Parameters:
	None
	
	Returns:
	C_values (np.array): Array of heat capacities
	D_values (np.array): Array of lattice sizes
	T_c (float): Critical temperature
	'''
	global config

	D_values = np.linspace(config['intial_lattice'], config['final_lattice'], config['lattice_step'])
	T_c = config['T_c']
	C_values = []
	print(f"\nIterating over different Lattice Sizes for Finite-Size Scaling of heat Capacity at Tc:\n")
	for D in tqdm(D_values, desc="Lattice Sizes"):
			_, C, _ = simulate(D, T_c)
			C_values.append(C)

	return C_values, D_values, T_c


def calculate_log_log(D_values, T_c, temperatures, magnetizations, susceptibilities, C_values):
	'''
	This function calculates the log-log values for Magnetization and Susceptibility to verify Onsager's Scaling Laws
	
	Parameters:
	D_values (np.array): Array of lattice sizes
	T_c (float): Critical temperature
	temperatures (np.array): Array of temperatures
	magnetizations (np.array): Array of magnetizations
	susceptibilities (np.array): Array of susceptibilities
	C_values (np.array): Array of heat capacities
	
	Returns:
	log_D (np.array): Log values of lattice sizes
	params (np.array): Parameters of the linear fit
	fit_C (np.array): Fitted values of heat capacity
	T_near_Tc (np.array): Array of temperatures near T_c
	M_near_Tc (np.array): Array of magnetizations near T_c
	Chi_near_Tc (np.array): Array of susceptibilities near T_c
	'''

	mask = (temperatures > T_c - 0.3) & (temperatures < T_c + 0.3)
	T_near_Tc = temperatures[mask]
	M_near_Tc = np.array(magnetizations)[mask]
	Chi_near_Tc = np.array(susceptibilities)[mask]

	#Log calculation for log x log plots

	log_D = np.log(D_values)
	params = np.polyfit(log_D, C_values, 1)
	fit_C = params[0] * log_D + params[1]

	return log_D, params, fit_C, T_near_Tc, M_near_Tc, Chi_near_Tc

#Finite Scaling Analysis for C vs ln(D) to prove proportionality

def plot_all(T_c, log_D, C_values, params, fit_C, T_near_Tc, M_near_Tc, Chi_near_Tc, temperatures, magnetizations, heat_capacities, susceptibilities):
	'''
	This function plots the results of the Ising Model simulation e.g. Finite-Size Scaling, Parameter vs Temperature, Log-Log plots, and Equilibrium visualization

	Parameters:
	T_c (float): Critical temperature
	log_D (np.array): Log values of lattice sizes
	C_values (np.array): Array of heat capacities
	params (np.array): Parameters of the linear fit
	fit_C (np.array): Fitted values of heat capacity
	T_near_Tc (np.array): Array of temperatures near T_c
	M_near_Tc (np.array): Array of magnetizations near T_c
	Chi_near_Tc (np.array): Array of susceptibilities near T_c
	temperatures (np.array): Array of temperatures
	magnetizations (np.array): Array of magnetizations
	heat_capacities (np.array): Array of heat capacities
	susceptibilities (np.array): Array of susceptibilities

	Returns:
	None
	'''

	plot_parameter_vs_temperature(T_c, temperatures, magnetizations, heat_capacities, susceptibilities)
	plot_finite_size_scaling(log_D, C_values, fit_C, params)
	plot_loglog(T_c, T_near_Tc, M_near_Tc, Chi_near_Tc)
	visualize_equilibrium()


def plot_finite_size_scaling(log_D, C_values, fit_C, params):
	'''
	This function plots the Finite-Size Scaling of Heat Capacity at T = T_c
	
	Parameters:
	log_D (np.array): Log values of lattice sizes
	C_values (np.array): Array of heat capacities
	fit_C (np.array): Fitted values of heat capacity
	params (np.array): Parameters of the linear fit

	Returns:
	None
	'''

	plt.figure(figsize=(20, 5))
	plt.xlabel('$\ln(D)$')
	plt.ylabel('$C$')
	#plt.legend()
	plt.title(f'Finite-Size Scaling at $T=T_c$ ($\\alpha=0$)')

	plt.plot(log_D, C_values, 's', label='Simulation')
	plt.plot(log_D, fit_C, '--', label=f'Fit: $C = {params[0]:.2f} \ln(D) + {params[1]:.2f}$')

	plt.show()


def plot_parameter_vs_temperature(T_c, temperatures, magnetizations, heat_capacities, susceptibilities):
	'''
	This function plots the trends of Magnetization, Heat Capacities, and Susceptibility with Temperature
	
	Parameters:
	T_c (float): Critical temperature
	temperatures (np.array): Array of temperatures
	magnetizations (np.array): Array of magnetizations
	heat_capacities (np.array): Array of heat capacities
	susceptibilities (np.array): Array of susceptibilities
	
	Returns:
	None
	'''

	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(33.333, 5), gridspec_kw = {'wspace': 0.4})

	# First subplot: Magnetization
	ax1.set_title('Magnetization')  # Add title
	ax1.axvline(T_c, color='red', linestyle='--', label='$T_c$ (Onsager)')
	ax1.set_xlabel("Temperature ($J/k_B$)")
	ax1.set_ylabel("⟨|M|⟩")
	ax1.plot(temperatures, magnetizations, 'o-', color='blue')
	ax1.legend()  # Optional: Show legend for the vertical line

	# Second subplot: Heat Capacity
	ax2.set_title('Heat Capacity')  # Add title
	ax2.axvline(T_c, color='red', linestyle='--')
	ax2.set_xlabel("Temperature ($J/k_B$)")
	ax2.set_ylabel("Heat Capacity $C$")
	ax2.plot(temperatures, heat_capacities, 'o-', color='green')

	# Third subplot: Susceptibility
	ax3.set_title('Susceptibility')  # Add title
	ax3.axvline(T_c, color='red', linestyle='--')
	ax3.set_xlabel("Temperature ($J/k_B$)")
	ax3.set_ylabel("Susceptibility $\\chi$")
	ax3.plot(temperatures, susceptibilities, 'o-', color='purple')

	plt.show()


# Log-log plots of Magnetization and Susceptibility to verify Onsager's Scaling Laws

def plot_loglog(T_c, T_near_Tc, M_near_Tc, Chi_near_Tc):
	'''
	This function plots the log-log values of Magnetization and Susceptibility to verify Onsager's Scaling Laws

	Parameters:
	T_c (float): Critical temperature
	T_near_Tc (np.array): Array of temperatures near T_c
	M_near_Tc (np.array): Array of magnetizations near T_c
	Chi_near_Tc (np.array): Array of susceptibilities near T_c

	Returns:
	None
	'''

	log_T = np.log(T_c - T_near_Tc[T_near_Tc < T_c])
	log_M = np.log(M_near_Tc[T_near_Tc < T_c])
	slope_M, intercept_M = np.polyfit(log_T, log_M, 1)

	log_T = np.log(T_near_Tc[T_near_Tc > T_c] - T_c)
	log_Chi = np.log(Chi_near_Tc[T_near_Tc > T_c])
	slope_Chi, intercept_Chi = np.polyfit(log_T, log_Chi, 1)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(33.333, 5), gridspec_kw = {'wspace': 0.4})

	# First subplot: Magnetization
	ax1.set_title('Magnetization Log')  # Add title
	ax1.set_xlabel('$\ln(T_c - T)$')
	ax1.set_ylabel('$\ln(M)$')
	ax1.plot(log_T, log_M, 'o', label='Data')
	ax1.plot(log_T, slope_M * log_T + intercept_M, '--', label=f'Fit: slope={slope_M:.3f} (β=1/8≈0.125)')
	ax1.legend()  # Optional: Show legend for the vertical line

	# Second subplot: Heat Capacity
	ax2.set_title('Susceptibility Log')  # Add title
	ax2.set_xlabel('$\ln(T - T_c)$')
	ax2.set_ylabel('$\ln(\chi)$')
	ax2.plot(log_T, log_Chi, 'o', label='Data')
	ax2.plot(log_T, slope_Chi * log_T + intercept_Chi, '--', label=f'Fit: slope={slope_Chi:.3f} (γ=-7/4≈-1.75)')

	plt.show()


def visualize_equilibrium():
	'''
	This function simulates the spin for a given D and T to be used for visualization during equilibrium
	
	Parameters:
	None
	
	Returns:
	None
	'''

	global config

	T = 1.0 #0.4, 2.269

	spins = initialize_spins(config['D'])

	spin_array = [spins.copy()]
	print("\nVisualizing Spins during Equilibrium\n")
	for _ in tqdm(range(500),desc="Equilibrium Step"):
		spins = metropolis_step(spins, T)
		spin_array.append(spins.copy())

#  _, _, _, spin_array = simulate(50, 2.269)
	plot_animation(spin_array)


def plot_animation(spin_array):
	'''
	This function plots the animation of the spins during equilibrium

	Parameters:
	spin_array (np.array): Array of spins
	
	Returns:
	None
	'''

	interval = 100       # Delay between frames (ms)

	# Initialize plot
	fig, ax = plt.subplots()  
	
	plt.xlabel(None)
	plt.ylabel(None )
	plt.xticks(color='white')
	plt.yticks(color='white')

	img = ax.imshow(spin_array[0] , cmap='binary', vmin=-1, vmax=1, interpolation='nearest')

	def update(frame):

		ax.set_title(f"2D Ising Model : Equilibriation Simulation - Step ({frame+1} of {len(spin_array)})")

		img.set_data(spin_array[frame])
		return img,

	ani = FuncAnimation(
			fig,
			update,
			frames = len(spin_array),
			interval = interval,
			blit = True,
			repeat = False
			)
	
	plt.show(block=False)
	plt.pause(len(spin_array) * interval / 1000)
	plt.close(fig)

if __name__ == "__main__":
    main()
