#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <semaphore.h>

#define MAX_PASSENGERS 10
#define TIME_COEFFICIENT 1000

int n, C, T, N;
FILE* fout;
char* out;
pthread_t car;
pthread_t passenger[MAX_PASSENGERS];
pthread_mutex_t mutex;
int passenger_time[MAX_PASSENGERS] = {0};
int car_time = 0;
int on_board[MAX_PASSENGERS] = {0};
int on_board_count = 0;
int ending = 0;
sem_t queue, checkin, boarding, riding, unloading;

void* PassengerThread(void* argv) {
    int passenger_id = *(int*)argv;
    while (!ending) {
        // Wander
        pthread_mutex_lock(&mutex);
        fprintf(fout, "Passenger %d wanders around the park.\n", passenger_id);
        pthread_mutex_unlock(&mutex);
        usleep(passenger_id * TIME_COEFFICIENT);
        passenger_time[passenger_id - 1] += passenger_id;

        // TakeRide
        pthread_mutex_lock(&mutex);
        fprintf(fout, "Passenger %d returns for another ride at %d millisec.\n", passenger_id, passenger_time[passenger_id - 1]);
        pthread_mutex_unlock(&mutex);
        sem_wait(&queue);
        sem_wait(&checkin);
        on_board[on_board_count] = passenger_id;
        on_board_count++;
        if (on_board_count == C) sem_post(&boarding);
        sem_post(&checkin);
        sem_wait(&riding);
        sem_post(&unloading);
    }
    pthread_exit(NULL);
}

void* CarThread() {
    int i, j;
    for (i = 0; i < N; i++) {
        // Wait
        for (j = 0; j < C; j++) {
            sem_post(&queue);
        }
        sem_wait(&boarding);
        if (car_time < passenger_time[on_board[C - 1] - 1]) car_time = passenger_time[on_board[C - 1] - 1];
        pthread_mutex_lock(&mutex);
        fprintf(fout, "Car departures at %d millisec. Passengers ", car_time);
        for (j = 0; j < C; j++) fprintf(fout, "%d  ", on_board[j]);
        fprintf(fout, "are in the car.\n");
        pthread_mutex_unlock(&mutex);

        // Ride
        usleep(T * TIME_COEFFICIENT);
        car_time += T;
        for (j = 0; j < C; j++) passenger_time[on_board[j] - 1] = car_time;
        pthread_mutex_lock(&mutex);
        fprintf(fout, "Car arrives at %d millisec. Passengers ", car_time);
        for (j = 0; j < C; j++) fprintf(fout, "%d  ", on_board[j]);
        fprintf(fout, "get off the car.\n");
        pthread_mutex_unlock(&mutex);
        if (i == N - 1) {
			ending = 1;
			for (j = 0; j < n; j++) {
				sem_post(&queue);
				sem_post(&checkin);
				sem_post(&riding);
			}
		}
		else {
			on_board_count = 0;
			for (j = 0; j < C; j++) {
				sem_post(&riding);
				sem_wait(&unloading);
			}
		}
    }
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    // Initialize
    n = atoi(argv[1]), C = atoi(argv[2]), T = atoi(argv[3]), N = atoi(argv[4]), out = argv[5];
	sem_init(&queue, 0, 0);
	sem_init(&checkin, 0, 1);
	sem_init(&boarding, 0, 0);
	sem_init(&riding, 0, 0);
	sem_init(&unloading, 0, 0);
    pthread_mutex_init(&mutex, NULL);
    fout = fopen(out, "w");
    fprintf(fout, "%d %d %d %d\n", n, C, T, N);

    // Run
    int id[n];
    int i;
    pthread_create(&car, NULL, &CarThread, NULL);
    for (i = 0; i < n; i++) {
        id[i] = i + 1;
        pthread_create(&passenger[i], NULL, &PassengerThread, (void*)&id[i]);
    }
    pthread_join(car, NULL);
	for (i = 0; i < n; i++) pthread_join(passenger[i], NULL);

    // Finalize
    fclose(fout);
    pthread_mutex_destroy(&mutex);
	sem_destroy(&queue), sem_destroy(&checkin), sem_destroy(&boarding), sem_destroy(&riding), sem_destroy(&unloading);
    pthread_exit(NULL);
    return 0;
}
