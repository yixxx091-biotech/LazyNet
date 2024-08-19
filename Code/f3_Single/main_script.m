% main_script.m
clear all;
clc;

% Define global variables
global beta time_points big_output_data;
beta = {
    [0.707, 0.707], % Beta for dx1/dt
    [0.555, -0.832], % Beta for dx2/dt
    [0.832, -0.555], % Beta for dx3/dt
    [0.600, -0.800], % Beta for dx4/dt
    [0.894, 0.447], % Beta for dx5/dt
    [0.894, 0], % Beta for dx6/dt
    [0.894, -0.447] % Beta for dx7/dt
};

big_output_data = [];
time_points = 0:0.1:49.9; % 99.9 for testset

% Generate and process samples
for i = 1:1
    initial_state = rand(1, 7);
    process_samples(initial_state);
end

% Save the results to a text file
filename = 'trainset062624_1.txt';
writematrix(big_output_data, filename);