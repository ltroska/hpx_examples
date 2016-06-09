import subprocess
import re

matrix_size = 8192
iterations = 10
max_processors = 4
init_points_per_block = 32
step = 32

outfile_path = 'matrix_vector_data_s' + str(matrix_size) + '_p' + str(max_processors) + '.dat'

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

points_per_block = init_points_per_block

data = []

while points_per_block <= matrix_size:
	num_blocks = matrix_size/points_per_block

	row = []

	for p in range(1, max_processors + 1):
		cmd = './build/matrix_vector_product --num_blocks ' + str(num_blocks) + ' --matrix_size ' + str(matrix_size) + ' -t' + str(p) + ' --iterations ' + str(iterations) + ' --hpx:print-counter=/threads{locality#*/total}/idle-rate'

		output = subprocess.check_output(cmd, shell=True)

		row.append(get_avg_time(output))
		row.append(get_rate(output))
		row.append(get_idle_rate(output))		
		
	print row
	data.append(row)

	points_per_block += step


#

with open(outfile_path, 'wb') as outfile:
	outfile.write('size ' + str(matrix_size) + '\n')
	outfile.write('iterations ' + str(iterations) + '\n')
	outfile.write('init_points_per_block ' + str(init_points_per_block) + '\n')
	outfile.write('step ' + str(step) + '\n')

	outfile.write('#processors ')
	for p in range(1, max_processors + 1):
		outfile.write(str(p) + ' ')

	outfile.write('\n')
	outfile.write('avg_time rate idle_rate\n')
	points_per_block = init_points_per_block

	i = 0
	while points_per_block <= matrix_size:
		outfile.write(str(points_per_block) + ' ')
		for value in data[i]:
			outfile.write(str(value) + ' ')
		outfile.write('\n')
		points_per_block += step
		i += 1
