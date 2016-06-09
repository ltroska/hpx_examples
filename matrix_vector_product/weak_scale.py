import subprocess
import re
import math

matrix_size_orig = 1024
iterations = 10
max_processors = 4
outfile_path = 'matrix_vector_data_p' + str(max_processors) + '_weak.dat'

def get_avg_time(output):
	pattern = 'Avg time \(s\):\t(.*?)\n'
	m = re.search(pattern, output)
	return m.group(1)

def get_idle_rate(output):
	pattern = '\/threads\{locality#0\/total\}\/idle-rate,.,[0-9\.]*,\[s\],([0-9]*),\[0.01%\]\n'
	m = re.search(pattern, output)
	return m.group(1) 

def get_rate(output):
	pattern = 'Rate \(MB\/s\):\t([0-9.]*?)\n'
	m = re.search(pattern, output)
	return m.group(1)


with open(outfile_path, 'wb') as outfile:
	outfile.write('size ' + str(matrix_size_orig) + '\n')
	outfile.write('iterations ' + str(iterations) + '\n')
	outfile.write('init_points_per_block ' + str(1) + '\n')

	outfile.write('#processors ')
	for p in range(1, max_processors + 1):
		outfile.write(str(p) + ' ')

	outfile.write('\n')
	outfile.write('avg_time rate idle_rate\n')

	for p in range(1, max_processors + 1):
		
		matrix_size = int(math.floor(math.sqrt(matrix_size_orig*matrix_size_orig*p)))
		
		num_blocks = matrix_size / p		

		print matrix_size, matrix_size / p, matrix_size / p * p

		cmd = './build/matrix_vector_product --num_blocks ' + str(num_blocks) + ' --matrix_size ' + str(matrix_size) + ' -t' + str(p) + ' --iterations ' + str(iterations) + ' --hpx:print-counter=/threads{locality#*/total}/idle-rate'

		output = subprocess.check_output(cmd, shell=True)
			
		outfile.write(str(p) + ' ' + get_avg_time(output) + ' ' + get_rate(output) + ' ' + get_idle_rate(output) + '\n')	
	


