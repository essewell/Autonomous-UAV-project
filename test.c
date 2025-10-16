#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main {
    char *str = "Hello, World!";
    char *copy = (char *)malloc(strlen(str) + 1);
    if (copy == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    strcpy(copy, str);
    printf("Original: %s\n", str);
    printf("Copy: %s\n", copy);
    free(copy);
    return 0;
}