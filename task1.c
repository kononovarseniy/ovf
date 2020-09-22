#include <stdio.h>
#include <math.h>

int main(int argc, char **argv) {
    double a = 1;
    while (1 + a / 2 > 1)
        a /= 2;
    printf("eps = %.13a %e\n", a, a);

    while (a / 2 > 0)
        a /= 2;
    printf("min = %.13a %e\n", a, a);

    while (a * 2 < INFINITY)
        a *= 2;
    printf("max = %.13a %e\n", a, a);
    return 0;
}
