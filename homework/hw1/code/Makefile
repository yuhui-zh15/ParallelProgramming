all:
	mpicc HW1_x1054034_advanced.c -o HW1_x1054034_advanced
	mpirun ./HW1_x1054034_advanced 21 testcase/testcase3 output
	./b2float output

build:
	mpicc HW1_x1054034_advanced.c -o HW1_x1054034_advanced

show:
	./b2float output

run1:
	mpirun ./HW1_x1054034_advanced 4 testcase/testcase1 output
	diff output ./testcase/sorted1

run3:
	mpirun ./HW1_x1054034_advanced 21 testcase/testcase3 output
	diff output ./testcase/sorted3

run9:
	mpirun ./HW1_x1054034_advanced 99999 testcase/testcase9 output
	diff output ./testcase/sorted9
