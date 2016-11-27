% error loglog plot for HW2
close all
clear variables
clc

derivative_error = load('../refinement_derivative_error.dat');
force_error = load('../refinement_force_error.dat');
epsilon = load('../refinement_epsilon.dat');

% This should give the SECOND order.
figure
plot(log(epsilon),log(derivative_error))
figure
plot(log(epsilon),log(force_error))