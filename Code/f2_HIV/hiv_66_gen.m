% hiv_66_gen.m
clear all
global sigma dt beta di f gamma k epison dp g de h cache time_points big_output_data
sigma = 1e6;
dt = 1;
beta = 1e-5;
di = 1;
f = [0.659605252908307; 0.518594942510538; 0.972974554763863; 0.648991492712356; 0.800330575352402; 0.453797708726920];
gamma = 1;
k = 4e-5;
epison = rand(6, 6);
cache = [0.01; 0.01; 0.01; 0.01; 0.01; 0.01];
epison = [epison, cache];
dp = 1;
g = 1.1;
de = 0.1;
h = 1e3;
time_points = 0:0.1:49.9;
initials = [1e6, 1, 1e8, 1, 1, 1, 1, 1, 1, 1, 1, 1e8, 1, 1, 1, 1, 1, 1, 1];

initial_state = initials;
big_output_data = [];

one_change_samples = generate_samples(initial_state, 1, 100);

for i = 1:100
    append_ode_results(one_change_samples(:, i), i);
end

sample_index = 101;
for num_changes = 2:19
    num_samples = round(100 * exp(-0.1 * (num_changes - 1)));
    multi_change_samples = generate_samples(initial_state, num_changes, num_samples);
    disp(num_samples);
    for j = 1:num_samples
        append_ode_results(multi_change_samples(:, j), sample_index);
        sample_index = sample_index + 1;
    end
end

filename = 'trainset.txt';
writematrix(big_output_data, filename);

function samples=generate_samples(base_state, n, num_samples)
    samples = zeros(length(base_state), num_samples);
    for i = 1:num_samples
        new_state = base_state;
        indices_to_change = randperm(length(base_state), n);
        new_state(indices_to_change) = 1 + (1e8 - 1).*rand(size(indices_to_change));
        samples(:, i) = new_state;
    end
end

function append_ode_results(initial_conditions, sample_index)
    global time_points big_output_data
    [t, N] = ode45(@hiv_66, time_points, initial_conditions);
    % Ensure the data has 501 time points, which is the correct count for 0:0.1:50
    if length(t) == 500
        output_data = [repmat(sample_index, length(t), 1), t, N]; % Concatenate results
        big_output_data = [big_output_data; output_data]; % Append to the big dataset
    end
end