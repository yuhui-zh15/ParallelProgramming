#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <termios.h>

#define HEIGHT 4
#define WIDTH 64
#define LEFT 97
#define RIGHT 100
#define UP 119
#define DOWN 115
#define FLUSH_TIME 0.5
#define DISPLAY_TIME 0.12
#define GAME_TIME 20
#define PER_SECONDS 1000000
int DIFFICULT = 5;

/* reads from keypress, doesn't echo */
int getch(void) {
    struct termios oldattr, newattr;
    int ch;
    tcgetattr( STDIN_FILENO, &oldattr );
    newattr = oldattr;
    newattr.c_lflag &= ~( ICANON | ECHO );
    tcsetattr( STDIN_FILENO, TCSANOW, &newattr );
    ch = getchar();
    tcsetattr( STDIN_FILENO, TCSANOW, &oldattr );
    return ch;
}

typedef struct { int velocity, direction, position, carlength, carnum; } Lane;

Lane lanes[HEIGHT];
int nowx = HEIGHT + 1;
int nowy = WIDTH / 2;
int status = 0;
int timeleft = GAME_TIME;

void Display() {
    char data[HEIGHT + 2][WIDTH];
    int i, j, k;
    for (i = 0; i < HEIGHT + 2; i++) {
        for (j = 0; j < WIDTH; j++) {
            data[i][j] = ' ';
        }
    }
    for (j = 0; j < WIDTH; j++) {
        data[0][j] = '-';
    }
    for (i = 0; i < HEIGHT; i++) {
        int spacing = WIDTH / lanes[i].carnum;
        for (j = 0; j < WIDTH; j++) {
            if ((j - lanes[i].position) % spacing == 0) {
                for (k = 0; k < lanes[i].carlength; k++) {
                    if (lanes[i].carlength > 3) data[i + 1][(j + k) % WIDTH] = '+';
					else data[i + 1][(j + k) % WIDTH] = '*';
                }
            }
        }
    }
    for (j = 0; j < WIDTH; j++) {
        data[HEIGHT + 1][j] = '-';
    }
    if (nowx != 0 && nowx != HEIGHT + 1 && data[nowx][nowy] != ' ') {
        status = -1;
        data[nowx][nowy] = 'X';
    }
    else {
        data[nowx][nowy] = 'O';
    }
    system("clear");
    printf("Time Left: %ds, Difficulty: %d\n", timeleft, DIFFICULT);
    printf("[W/A/S/D] Direction, [R] Reset, [N/B] Difficulty, [Q] Quit\n");
    for (i = 0; i < HEIGHT + 2; i++) {
        for (j = 0; j < WIDTH; j++) {
			if (data[i][j] == '-') printf("🌫");
			else if (data[i][j] == 'O') printf("🐸");
			else if (data[i][j] == 'X') printf("🐷");
			else if (data[i][j] == '+') printf("🚋");
			else if (data[i][j] == '*') printf("🚐");
			else if (data[i][j] == ' ') printf(" ");
        }
        printf("\n");
    }
}

void* DisplayThread() {
    while (!status) {
        Display();
        usleep(DISPLAY_TIME * PER_SECONDS);
    }
    pthread_exit(NULL);
}

void* Update(void* argv) {
    Lane* lane = (Lane*)argv;
    while (!status) {
        usleep(FLUSH_TIME * PER_SECONDS / lane->velocity);
        if (lane->direction == LEFT) {
            lane->position = lane->position - 1;
            if (lane->position == -1) lane->position = WIDTH - 1;
        }
        else if (lane->direction == RIGHT) {
            lane->position = lane->position + 1;
            if (lane->position == WIDTH) lane->position = 0;
        }
    }
    pthread_exit(NULL);
}

void* TimeThread() {
    while (timeleft && !status) {
        usleep(PER_SECONDS);
        timeleft--;
    }
    if (!status) status = 2;
    pthread_exit(NULL);
}

void Init() {
    int i;
	timeleft = GAME_TIME;
	nowx = HEIGHT + 1;
	nowy = WIDTH / 2;
    for (i = 0; i < HEIGHT; i++) {
        lanes[i].velocity = rand() % DIFFICULT + 1;
        lanes[i].carlength = rand() % DIFFICULT + 1;
        lanes[i].direction = (rand() % 2)? LEFT: RIGHT;
        lanes[i].position = rand() % WIDTH;
        lanes[i].carnum = WIDTH / (lanes[i].carlength + (15 - DIFFICULT));
    }
}

int main(int argc, char** argv) {
	if (argc == 2) DIFFICULT = atoi(argv[1]);
    srand((unsigned)time(NULL));
    Init();
    pthread_t threads[HEIGHT];
    pthread_t displaythread;
    pthread_t timethread;
	int i;
    for (i = 0; i < HEIGHT; i++) {
        pthread_create(&threads[i], NULL, &Update, (void*)&lanes[i]);
    }
    pthread_create(&displaythread, NULL, &DisplayThread, NULL);
    pthread_create(&timethread, NULL, &TimeThread, NULL);
    while (!status) {
        char op = getch();
        switch(op) {
        case LEFT:
            nowy = nowy - 1;
            if (nowy == -1) nowy = WIDTH - 1;
            continue;
        case RIGHT:
            nowy = nowy + 1;
            if (nowy == WIDTH) nowy = 0;
            continue;
        case UP:
            nowx = nowx - 1;
            if (nowx == 0) status = 1;
            continue;
        case DOWN:
            nowx = nowx + 1;
            if (nowx == HEIGHT + 2) nowx = HEIGHT + 1;
            continue;
		case 'q':
			status = 3;
			break;	
		case 'r':
			Init();
			continue;
		case 'n':
			if(DIFFICULT <= 9) DIFFICULT += 1;
			Init();
			continue;
		case 'b':
			if (DIFFICULT >= 2) DIFFICULT -= 1;
			Init();
			continue;
		}
    }
    Display();
    if (status == 1) printf("WIN\n");
    else if (status == -1) printf("LOSE\n");
    else if (status == 2) printf("TIME LIMIT\n");
	else if (status == 3) printf("EXIT\n");
    pthread_exit(NULL);
    return 0;
}
