function dxdt = GRN_model(t, x, beta)
global beta
    % Initialize the derivative vector
    dxdt = zeros(7,1);
    
    % Define the ODEs based on the given equations
    dxdt(1) = 0.05 * (beta{1}(1) * x(1) + beta{1}(2) * x(2));
    dxdt(2) = cos(beta{2}(1) * x(2) + beta{2}(2) * x(3));
    dxdt(3) = sin(beta{3}(1) * x(2) + beta{3}(2) * x(3));
    dxdt(4) = 0.1 * (beta{4}(1) * x(2) + beta{4}(2) * x(4));
    dxdt(5) = sin(beta{5}(1) * x(2) + beta{5}(2) * x(4));
    dxdt(6) = 0.05 * exp(beta{6}(1) * x(3) + beta{6}(2) * x(2));
    dxdt(7) = 0.2 * (beta{7}(1) * x(2) + beta{7}(2) * x(3));