import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
	print "Usage: " + sys.argv[0] + " <filename>"
	sys.exit(1)

block_sizes = []
runtimes = []
rates = []
idle_rates = []

size = 0
iterations = 0

infile_path = sys.argv[1]

markers = ['o', 'x', '+', '*']
colors = ['r', 'b', 'g', 'k']

with open(infile_path, 'rb') as infile:
	size = infile.readline().split()[1]
	iterations = infile.readline().split()[1]
	init_points_per_block = infile.readline().split()[1]
	step = infile.readline().split()[1]
	processors = infile.readline().split(' ', 1)[1].split()
	infile.readline()
	
	for _ in range(len(processors)):
		runtimes.append([])
		rates.append([])
		idle_rates.append([])

	for line in infile:
		splitline = line.split(' ')
		block_sizes.append(splitline[0])


		for p in range(len(processors)):
			runtimes[p].append(splitline[1 + p * 3])
			rates[p].append(splitline[2 + p * 3])
			idle_rates[p].append(float(splitline[3 + p * 3])/100)


outfile = infile_path[:-4]

for i, p in enumerate(processors):
	plt.loglog(block_sizes, runtimes[i], label=p + ' thread(s)',  marker=markers[i], color=colors[i])

plt.title('Runtime for a ' + size + 'x' + size + ' matrix averaged over ' + iterations + ' iterations', y=1.08) 
plt.xlabel('Block Size')
plt.ylabel('Runtime (s)')
plt.grid(True, which='both')
plt.legend(loc='upper center')
plt.tight_layout()
plt.savefig(outfile + '_runtime')
plt.clf()

for i, p in enumerate(processors):
	plt.plot(block_sizes, rates[i], label=p + ' thread(s)',  marker=markers[i], color=colors[i])

ax = plt.gca()
ax.set_xscale('log')
plt.title('Rate for a ' + size + 'x' + size + ' matrix averaged over ' + iterations + ' iterations', y=1.08) 
plt.xlabel('Block Size')
plt.ylabel('Rate (MB/s)')
plt.grid(True, which='both')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(outfile + '_rate')
plt.clf()

for i, p in enumerate(processors):
	plt.plot(block_sizes, idle_rates[i], label=p + ' thread(s)',  marker=markers[i], color=colors[i])

ax = plt.gca()
ax.set_xscale('log')
plt.title('Average Idle Rate for a ' + size + 'x' + size + ' matrix averaged over ' + iterations + ' iterations', y=1.08) 
plt.xlabel('Block Size')
plt.ylabel('Idle Rate (%)')
plt.grid(True, which='both')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(outfile + '_idle_rate')
