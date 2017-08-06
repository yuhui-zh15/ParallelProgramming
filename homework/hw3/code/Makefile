CC			:= gcc
CXX			:= g++

LDFLAGS			:= -lpthread 

all: roller frogger frogger_emoji

roller: HW3_roller_x1054034.c
	$(CC) $(LDFLAGS) -o $@ $?
frogger: HW3_frogger_x1054034.c
	$(CC) $(LDFLAGS) -o $@ $?
frogger_emoji: HW3_frogger_x1054034_emoji.c
	$(CC) $(LDFLAGS) -o $@ $?
clean:
	rm -f roller frogger frogger_emoji
