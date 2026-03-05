#include <math.h>
#include <algorithm>


double factorial(double n) {return (n<.5) ? 1.0 : n*factorial(n-1);}

double wigner3j(double a, double b, double c, double alpha, double beta, double gamma)
{
	double tri = factorial(a+b-c)*factorial(a-b+c)*factorial(-a+b+c)/factorial(a+b+c+1);
	double f = factorial(a+alpha)*factorial(a-alpha)*factorial(b+beta)*factorial(b-beta)*factorial(c+gamma)*factorial(c-gamma);

	double t_up[3] = {a+b-c,a-alpha,b+beta};
	double t_lo[3] = {-alpha+b-c,a+beta-c,0};
	int t_max = *std::min_element(t_up,t_up+3);
	int t_min = *std::max_element(t_lo,t_lo+3);

	double x = 0;
	int t;
	for (t=t_min; t<=t_max; t++) {
		x += pow(-1,t)/(factorial(t-t_lo[0])*factorial(t-t_lo[1])*factorial(t_up[0]-t)*factorial(t_up[1]-t)*factorial(t_up[2]-t));
	}
	return pow(-1,a-b-gamma)*sqrt(tri*f)*x;

}

// Reference: Wigner 3j-Symbol entry of Eric Weinstein's Mathworld: http://mathworld.wolfram.com/Wigner3j-Symbol.html


extern "C" {
double clebsch_gordan(double j1, double j2, double j, double m1, double m2, double m)
{
	// error checking or leave it unsafe for speed
	if ((m1+m2) != m) {
		return 0.0;
	} else {
		return pow(-1,j1-j2+m)*sqrt(2*j+1)*wigner3j(j1,j2,j,m1,m2,-m);
	}
}
}

// gcc -c clebsch_gordan_src.cpp
// gcc -shared -o libcg.dll clebsch_gordan_src.o
