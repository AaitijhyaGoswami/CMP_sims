function dynamic_localization_2d()
    % Interactive 2D simulation of a charged particle on a lattice.
    % Adjust the slider to change the Field/Frequency ratio
    % Physics based on: Dunlap and Kenkre, Phys. Rev. B 34, 3625 (1986).
    
    % SYSTEM PARAMETERS
    N = 41;                 % Lattice size (NxN). Kept small for speed.
    center_idx = (N+1)/2;
    V = 1;                  % Hopping integral
    hbar = 1;
    omega = 5;              % Fixed frequency
    t_max = 10;             % Duration of each simulation run
    dt = 0.05;              % Time step for animation
    
    % Creating the figure window
    fig = figure('Name', '2D Dynamic Localization', 'Color', 'w', ...
                 'Position', [100, 100, 900, 700], 'NumberTitle', 'off');

    % Slider for Ratio (E/omega)
    uicontrol('Style', 'text', 'Position', [100, 50, 150, 20], ...
              'String', 'Field Ratio (E/\omega)', 'FontSize', 10, 'BackgroundColor', 'w');
    
    ratio_slider = uicontrol('Style', 'slider', 'Min', 0, 'Max', 5, 'Value', 0, ...
                             'Position', [100, 30, 200, 20], ...
                             'Callback', @start_simulation);
                         
    val_label = uicontrol('Style', 'text', 'Position', [310, 30, 50, 20], ...
                          'String', '0.00', 'FontSize', 10, 'BackgroundColor', 'w');

    % Run Button
    uicontrol('Style', 'pushbutton', 'String', 'Run Simulation', ...
              'Position', [400, 30, 120, 30], 'FontSize', 10, ...
              'Callback', @start_simulation);
          
    % Infomative Text
    uicontrol('Style', 'text', 'Position', [550, 20, 300, 40], ...
              'String', 'Note: Ratio ~2.405 causes localization.', ...
              'FontSize', 9, 'BackgroundColor', 'w', 'HorizontalAlignment', 'left');

    % Axes for 2D Plot
    ax = axes('Parent', fig, 'Position', [0.15, 0.2, 0.7, 0.7]);
    
    % SIM LOGIC
    function start_simulation(~, ~)
        % Getting parameters from GUI
        ratio = get(ratio_slider, 'Value');
        set(val_label, 'String', num2str(ratio, '%.2f'));
        E_mag = ratio * omega;
        
        % We will apply the field at 45 degrees to show anisotropy clearly
        % E = (E_x, E_y). Let's set E_x = E_mag and E_y = 0 to show 
        % "pancake" shapes (localized in X, delocalized in Y)
        % OR set E_x = E_y = E_mag/sqrt(2) for diagonal effects.
        % For clearest demo of Eq (3.5) in paper:
        % Let's apply field ONLY along X-axis. 
        % This effectively makes V_eff_x = V*J0(ratio) and V_eff_y = V.
        E_vec = [E_mag, 0]; 

        % Initial State: Localized at center
        psi = zeros(N, N);
        psi(center_idx, center_idx) = 1;
        psi = psi(:); % Flatten to vector
        
        % Pre-computing Hamiltonian parts
        % 1D kinetic matrices for Kronecker sum
        e = ones(N, 1);
        H_1D_kin = spdiags([V*e V*e], [-1 1], N, N);
        I = speye(N);
        % 2D Kinetic Energy: H_x + H_y
        H_kin = kron(H_1D_kin, I) + kron(I, H_1D_kin);
        
        % Potential Energy Operators (Diagonal)
        % X and Y coordinate matrices
        x_indices = ((1:N) - center_idx)';
        X_op = spdiags(x_indices, 0, N, N);
        Y_op = spdiags(x_indices, 0, N, N); % Same shape, different Kronecker slot
        
        % Full Potential Operator: -e(E_x*X + E_y*Y)
        % Note: E is time dependent, so we build the static part here
        % H_pot(t) = -cos(wt) * (E_x * X_2D + E_y * Y_2D)
        term_x = kron(X_op, I);
        term_y = kron(I, Y_op);
        Pot_static_x = -E_vec(1) * term_x;
        Pot_static_y = -E_vec(2) * term_y;

        % Time Loop
        t = 0;
        
        cla(ax);
        [X, Y] = meshgrid(1:N, 1:N);
        h_surf = surf(ax, X, Y, reshape(abs(psi).^2, N, N));
        view(ax, 2); % Top-down view
        shading(ax, 'interp');
        colormap(ax, 'jet');
        c = colorbar(ax);
        c.Label.String = 'Probability Density';
        axis(ax, [1 N 1 N]);
        title(ax, sprintf('Time: %.2f | Ratio: %.2f', t, ratio));
        xlabel(ax, 'X (Field Direction)');
        ylabel(ax, 'Y (Free Direction)');
        
        % Simulation Loop
        while t < t_max
            % Create H(t) for this step
            % H(t) = H_kin + cos(omega*t) * (Pot_static_x + Pot_static_y)
            field_factor = cos(omega * t);
            H_total = H_kin + field_factor * (Pot_static_x + Pot_static_y);
            
            % Simple Euler/Crank-Nicolson step or just matrix exponentiation for small dt
            % For animation speed, we use a first-order approximation or small ODE step
            % psi(t+dt) = expm(-i*H*dt)*psi(t). 
            % Note: expm is slow for large sparse matrices. 
            % Using RK4 is better for speed here.
            
            k1 = -1i/hbar * (H_total * psi);
            
            % Approximating H as constant over dt for k2, k3, k4 (valid for small dt)
            k2 = -1i/hbar * (H_total * (psi + 0.5*dt*k1));
            k3 = -1i/hbar * (H_total * (psi + 0.5*dt*k2));
            k4 = -1i/hbar * (H_total * (psi + dt*k3));
            
            psi = psi + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
            
            % Updating Plot
            prob_density = reshape(abs(psi).^2, N, N);
            set(h_surf, 'ZData', prob_density);
            title(ax, sprintf('Time: %.2f | Ratio: %.2f', t, ratio));
            drawnow;
            
            t = t + dt;
            
            % Checking if figure was closed
            if ~isvalid(fig), break; end
        end
    end
end