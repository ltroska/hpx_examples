import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 4:
	print "Usage: " + sys.argv[0] + " <filename hpx> <filename mpi> <0=weak/1=strong>"
	sys.exit(1)

weak = sys.argv[3] == '0'
title = ''

if weak:
	title = 'Weak'
else:
	title = 'Strong'

runtimes_hpx_orig = []
runtimes_mpi_orig = []
rates = []
idle_rates = []
processors = []

size = 0
iterations = 0

infile_path_hpx = sys.argv[1]
infile_path_mpi = sys.argv[2]

markers = ['o', 'x', '+', '*']
colors = ['r', 'b', 'g', 'k']

with open(infile_path_hpx, 'rb') as infile:
	size = infile.readline().split()[1]
	iterations = infile.readline().split()[1]
	init_points_per_block = infile.readline().split()[1]
	infile.readline()
	infile.readline()

	for line in infile:
		splitline = line.split(' ')
		processors.append(splitline[0])

		runtimes_hpx_orig.append(splitline[1])
		rates.append(splitline[2])
		idle_rates.append(float(splitline[3])/100)


outfile = infile_path_hpx[:-4]

hpx_t0 = float(runtimes_hpx_orig[0])

runtimes_hpx = [float(t) / hpx_t0 for t in runtimes_hpx_orig]

plt.plot(processors, runtimes_hpx, label='HPX',  marker=markers[0], color=colors[0])
plt.title(title + ' scaling for a ' + size + 'x' + size + ' matrix, averaged over ' + iterations + ' iterations and normalized', y=1.08)
plt.xlabel('#Processors')
plt.ylabel('Runtime (s)')
plt.grid(True, which='both')
plt.legend(loc='upper center')
plt.tight_layout()
plt.savefig(outfile + '_scaling')
plt.clf()

speedup_hpx = [1 / float(t) for t in runtimes_hpx]
speedup_ideal = [(p+1) for p in range(len(processors))]

plt.plot(processors, speedup_hpx, label='HPX',  marker=markers[0], color=colors[0])
plt.plot(processors, speedup_ideal, label='ideal',  marker=markers[1], color=colors[1])
plt.title(title + ' speedup for a ' + size + 'x' + size + ' matrix, averaged over ' + iterations + ' iterations', y=1.08)
plt.xlabel('#Processors')
plt.ylabel('Speedup')
plt.grid(True, which='both')
plt.legend(loc='upper center')
plt.tight_layout()
plt.savefig(outfile + '_speedup')
plt.clf()

efficiency_hpx = [s / (p + 1) for p, s in enumerate(speedup_hpx)]

plt.plot(processors, efficiency_hpx, label='HPX',  marker=markers[0], color=colors[0])
plt.title(title + ' efficiency for a ' + size + 'x' + size + ' matrix, averaged over ' + iterations + ' iterations', y=1.08)
plt.xlabel('#Processors')
plt.ylabel('Efficiency')
plt.grid(True, which='both')
plt.legend(loc='upper center')
plt.tight_layout()
plt.savefig(outfile + '_efficiency')
plt.clf()
