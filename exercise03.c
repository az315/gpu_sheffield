//
// Created by az313 on 15.03.2025.
//

/*#include <stdio.h>
#include <stdlib.h>
#include "linked_list.h"

#define NUM_STUDENTS 4

struct student{
    char *forename;
    char *surname;
    float average_module_mark;

};

void print_student(const struct student *s);

void main(){
    llitem *first;
    llitem *end;
    llitem *ll;
    unsigned int str_len;
    // struct student *students;
    // students = (struct student*) malloc(NUM_STUDENTS * sizeof(struct student));

    first = NULL;
    end = NULL;
    print_callback = (void(*)(void *))&print_student;

    FILE *f = NULL;
    f = fopen("C:/Users/az313/development/gpu-sheffield/students2.bin", "rb"); //read and binary flags
    if (f == NULL){
        fprintf(stderr, "Error: Could not find students2.bin file \n");
        exit(1);
    }

    while (fread(&str_len, sizeof(unsigned int), 1, f) == 1) {
        struct student *s;
        s = (struct student*) malloc(sizeof(struct student));
        s->surname = (char*) malloc(sizeof(char) * str_len);
        s->forename = (char*) malloc(sizeof(char) * str_len);
        fread(s->surname, sizeof(char), str_len, f);
        fread(s->forename, sizeof(char), str_len, f);
        fread(&s->average_module_mark, sizeof(float), 1, f);

        if (end == NULL) {
            end = create_linked_list();
            first = end;
        } else {
            end = add_to_linked_list(end);
        }

        end->record = (void*)s;
    }

    fclose(f);
    print_items(first);

    ll = first;
    while (ll != NULL) {
        free(((struct student*)ll->record)->forename);
        free(((struct student*)ll->record)->surname);
        free(ll->record);
        ll = ll->next;
    }
    free_linked_list(first);
}

void print_student(const struct student *s){
    printf("Student:\n");
    printf("\tForename: %s\n", s->forename);
    printf("\tSurname: %s\n", s->surname);
    printf("\tAverage Module Mark: %.2f\n", s->average_module_mark);
}*/
