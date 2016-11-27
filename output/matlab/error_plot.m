% error loglog plot for HW2
clear variables
clc

derivative_error = load('../refinement_derivative_error.dat');
epsilon = load('../refinement_epsilon.dat');

% This should give the SECOND order.
plot(log(epsilon),log(derivative_error))
