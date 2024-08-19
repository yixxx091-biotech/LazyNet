% batch_functions.m

function process_samples(initial_state)
    global beta;
    
    % Example of processing samples in a batch
    for num_changes = 1:length(beta)
        num_samples = round(100 * exp(-0.1 * (num_changes - 1)));
        samples = generate_samples(initial_state, num_changes, num_samples);
        
        for i = 1:num_samples
            append_ode_results(samples(:, i), i);
        end
    end
end

function samples = generate_samples(base_state, n, num_samples)
    % Generate samples with n changes
    samples = zeros(length(base_state), num_samples);
    for i = 1:num_samples
        new_state = base_state;
        indices_to_change = randperm(length(base_state), n);
        new_state(indices_to_change) = rand(size(indices_to_change));
        samples(:, i) = new_state;
    end
end

function append_ode_results(initial_conditions, sample_index)
    global time_points big_output_data;
    
    % Solve ODE
    [t, N] = ode45(@GRN_model, time_points, initial_conditions);
    
    % Append results to global data
    if length(t) == 500 %change to 1000 for testset
        output_data = [repmat(sample_index, length(t), 1), t, N];
        big_output_data = [big_output_data; output_data];
    end
end