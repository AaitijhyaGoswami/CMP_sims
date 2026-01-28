function floquet()
    % Parameters based on Dunlap and Kenkre (1986)
    % System properties
    N = 101;            % Lattice size (odd to have a distinct center)
    center_site = (N+1)/2;
    V = 1;              % Nearest-neighbor hopping integral [cite: 18]
    hbar = 1;           % Reduced Planck's constant (set to 1 as in paper [cite: 29])
    
    % Time parameters
    t_max = 20;         % Maximum simulation time (dimensionless Vt)
    t_span = linspace(0, t_max, 300);
    
    % Define the simulation cases (Ratio = E_magnitude / omega)
    % Case 1: No field (Delocalized)
    % Case 2: AC Field with "Magic Ratio" (Localized) -> First root of J0 ~ 2.405 [cite: 135]
    % Case 3: AC Field with Non-magic Ratio (Delocalized)
    
    cases = struct('name', {}, 'ratio', {}, 'omega', {});
    cases(1) = struct('name', 'No Field (E=0)', 'ratio', 0, 'omega', 1);
    cases(2) = struct('name', 'Localized (Ratio \approx 2.405)', 'ratio', 2.4048, 'omega', 5);
    cases(3) = struct('name', 'Delocalized (Ratio = 1.5)', 'ratio', 1.5, 'omega', 5);

    % Initialize figure
    figure('Color', 'w', 'Position', [100, 100, 1000, 600]);
    
    for i = 1:length(cases)
        % Field parameters
        % From paper: ratio = E_mag / omega. 
        % Therefore E_mag = ratio * omega.
        omega = cases(i).omega;
        E_mag = cases(i).ratio * omega; 
        
        % Initial State: Particle localized at the center site m=0
        psi0 = zeros(N, 1);
        psi0(center_site) = 1;
        
        % Solve ODE
        % We solve d(psi)/dt = -1i/hbar * H(t) * psi
        [t, psi_t] = ode45(@(t, psi) hamiltonian_func(t, psi, N, V, E_mag, omega, hbar), t_span, psi0);
        
        % Calculate Observables
        prob_density = abs(psi_t).^2;
        
        % Mean Square Displacement <m^2> 
        % Sites are indexed 1 to N. We shift them so center is 0.
        sites = ((1:N) - center_site)'; 
        msd = zeros(length(t), 1);
        
        for k = 1:length(t)
            msd(k) = sum((sites.^2) .* prob_density(k, :)');
        end
        
        % Plotting MSD
        subplot(1, 2, 1);
        hold on;
        plot(t, msd, 'LineWidth', 2, 'DisplayName', cases(i).name);
        
        % Plotting Probability Distribution at final time
        subplot(1, 2, 2);
        hold on;
        plot(sites, prob_density(end, :), 'LineWidth', 1.5, 'DisplayName', cases(i).name);
    end
    
    % Format Plots
    subplot(1, 2, 1);
    title('Mean Square Displacement \langle m^2 \rangle');
    xlabel('Time (t)');
    ylabel('\langle m^2 \rangle');
    legend('Location', 'northwest');
    grid on;
    % Replicate paper observation: Bounded MSD for Ratio=2.405 [cite: 136]
    
    subplot(1, 2, 2);
    title(['Probability Distribution at t = ' num2str(t_max)]);
    xlabel('Lattice Site m');
    ylabel('Probability |\psi_m|^2');
    xlim([-20 20]); % Zoom in on center
    legend('Location', 'northeast');
    grid on;

end

% ---------------------------------------------------------
% Helper Function: Time-Dependent Hamiltonian
% ---------------------------------------------------------
function dpsidt = hamiltonian_func(t, psi, N, V, E_mag, omega, hbar)
    % Construct the Hamiltonian matrix H(t)
    
    % 1. Off-diagonal elements (Hopping V) [cite: 18]
    % H includes |m><m+1| and |m+1><m|
    e = ones(N, 1);
    H_kin = spdiags([V*e V*e], [-1 1], N, N);
    
    % 2. Diagonal elements (Electric Potential)
    % The paper Hamiltonian term is -eE(t)a * sum(m |m><m|) [cite: 19]
    % Here we define E(t) = E_mag * cos(omega * t).
    % Sites are shifted so the center is m=0.
    m_indices = ((1:N) - (N+1)/2)';
    
    % The paper defines parameter epsilon = eEa. 
    % We treat E_mag as this epsilon parameter.
    field_strength = E_mag * cos(omega * t); 
    
    % Potential energy term on diagonal
    H_pot = spdiags(-field_strength * m_indices, 0, N, N);
    
    % Total Hamiltonian
    H = H_kin + H_pot;
    
    % Schrödinger Equation: d(psi)/dt = -i/hbar * H * psi
    dpsidt = -1i / hbar * H * psi;
end