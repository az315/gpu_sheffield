//
// Created by az313 on 15.03.2025.
//
/*#include <stdio.h>
#include <stdlib.h>

#define NUM_STUDENTS 4

struct student{
    char *forename;
    char *surname;
    float average_module_mark;

};

void print_student(const struct student *s);

void main(){
    struct student *students;
    students = (struct student*) malloc(NUM_STUDENTS * sizeof(struct student));

    int i;

    FILE *f = NULL;
    f = fopen("C:/Users/az313/development/gpu-sheffield/students2.bin", "rb"); //read and binary flags
    if (f == NULL){
        fprintf(stderr, "Error: Could not find students.bin file \n");
        exit(1);
    }

    fread(students, sizeof(struct student), NUM_STUDENTS, f);


    for (i = 0; i < NUM_STUDENTS; i++){
        unsigned int str_len;

        str_len = fread(&str_len, sizeof(unsigned int), 1, f);
        students[i].forename = (char*) malloc(sizeof(char) * str_len);
        fread(students[i].forename, sizeof(char), 1, f);

        str_len = fread(&str_len, sizeof(unsigned int), 1, f);
        students[i].surname = (char*) malloc(sizeof(char) * str_len);
        fread(students[i].surname, sizeof(char), 1, f);

        fread(&(students[i].average_module_mark), sizeof(float), 1, f);
    }
    fclose(f);

    for (i = 0; i < NUM_STUDENTS; i++) {
        print_student(&students[i]);
    }

    for (i = 0; i < NUM_STUDENTS; i++) {
        free(students[i].forename);
        free(students[i].surname);
    }

    free(students);
}

void print_student(const struct student *s){
    printf("Student:\n");
    printf("\tForename: %s\n", s->forename);
    printf("\tSurname: %s\n", s->surname);
    printf("\tAverage Module Mark: %.2f\n", s->average_module_mark);
}*/

