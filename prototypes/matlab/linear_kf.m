% Loads reference data exported by prototypes/python/linear_kf.py and compares hand-coded filter output to Python.

function linear_kf()
    thisDir = fileparts(mfilename('fullpath'));
    refPath = fullfile(thisDir, '..', 'output', 'linear_kf_ref.mat');
    if ~isfile(refPath)
        error('linear_kf:MissingRef', ...
            'Reference file not found. Run prototypes/python/linear_kf.py first.\n  Expected: %s', refPath);
    end

    ref = load(refPath);
    dt = ref.dt;
    F = ref.F;
    H = ref.H;
    Q = ref.Q;
    R = ref.R;
    z = ref.z;
    x_true = ref.x_true;
    x_est_py = ref.x_est_py;
    P0 = ref.P0;

    n = size(z, 1);
    x0 = initialStateFromMeasurements(z, dt);

    [x_est, P_hist] = runFilter(z, F, H, Q, R, x0, P0, dt);

    maxErr = max(abs(x_est(:) - x_est_py(:)));
    tol = 1e-9;
    fprintf('Max |x_mat - x_py|: %.3e (tol %.1e)\n', maxErr, tol);
    if maxErr > tol
        error('linear_kf:ParityFailed', 'MATLAB and Python trajectories differ beyond tolerance.');
    end
    fprintf('Parity check: PASS\n');

    nees = neesPerStep(x_true, x_est, P_hist);
    fprintf('Mean NEES (all steps): %.3f\n', mean(nees));

  % Optional trackingKF comparison
    compareTrackingKF(z, F, H, Q, R, P0, dt, x_est);

    plotTrajectory(z, x_true, x_est, P_hist, thisDir);
end

function x0 = initialStateFromMeasurements(z, dt)
    v0 = (z(2, :) - z(1, :)) / dt;
    x0 = [z(1, 1); z(1, 2); v0(1); v0(2)];
end

function [x_est, P_hist] = runFilter(z, F, H, Q, R, x0, P0, dt)
    n = size(z, 1);
    stateDim = 4;
    x = x0(:);
    P = P0;
    x_est = zeros(n, stateDim);
    P_hist = zeros(stateDim, stateDim, n);

    x_est(1, :) = x.';
    P_hist(:, :, 1) = P;

    for k = 2:n
        [x, P] = kfPredict(x, P, F, Q);
        [x, P] = kfUpdate(x, P, z(k, :).', H, R);
        x_est(k, :) = x.';
        P_hist(:, :, k) = P;
    end
end

function [x, P] = kfPredict(x, P, F, Q)
    x = F * x;
    P = F * P * F.' + Q;
end

function [x, P, y, S, K] = kfUpdate(x, P, z, H, R)
    y = z - H * x;
    S = H * P * H.' + R;
    K = P * H.' / S;
    x = x + K * y;
    I = eye(size(P, 1));
    P = (I - K * H) * P * (I - K * H).' + K * R * K.';
end

function nees = neesPerStep(x_true, x_est, P_hist)
    n = size(x_true, 1);
    nees = zeros(n, 1);
    for k = 1:n
        err = (x_true(k, :) - x_est(k, :)).';
        Pk = P_hist(:, :, k);
        nees(k) = err.' * (Pk \ err);
    end
end

function compareTrackingKF(z, F, H, Q, R, P0, dt, x_est_hand)
    if ~exist('trackingKF', 'file')
        fprintf('trackingKF not available — skipping toolbox comparison.\n');
        return;
    end

    n = size(z, 1);
    x0 = initialStateFromMeasurements(z, dt);

    try
        % trackingKF is linear: pass F,H as matrices (not EKF-style Jacobians).
        % MotionModel becomes "Custom"; F already embeds dt from the reference export.
        kf = trackingKF(F, H, ...
            'State', x0, ...
            'StateCovariance', P0, ...
            'ProcessNoise', Q, ...
            'MeasurementNoise', R);

        x_tkf = zeros(n, 4);
        x_tkf(1, :) = kf.State(:).';
        for k = 2:n
            predict(kf);
            correct(kf, z(k, :).');
            x_tkf(k, :) = kf.State(:).';
        end

        posErr = max(abs(x_tkf(:, 1:2) - x_est_hand(:, 1:2)), [], 'all');
        fprintf(['trackingKF vs hand-coded max position error: %.3f m\n', ...
            '  (differences expected: toolbox may use a different Q parameterization)\n'], posErr);
    catch me
        fprintf('trackingKF comparison failed: %s\n', me.message);
    end
end

function plotTrajectory(z, x_true, x_est, P_hist, thisDir)
    outDir = fullfile(thisDir, '..', 'output');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    figure('Visible', 'off');
    hold on;
    plot(x_true(:, 1), x_true(:, 2), 'k-', 'LineWidth', 1.5);
    scatter(z(:, 1), z(:, 2), 8, 'filled', 'MarkerFaceAlpha', 0.35);
    plot(x_est(:, 1), x_est(:, 2), 'b-', 'LineWidth', 1.5);
    axis equal;
    grid on;
    xlabel('x [m]');
    ylabel('y [m]');
    legend('Ground truth', 'Measurements', 'KF estimate', 'Location', 'best');
    title('Linear CV KF (MATLAB)');
    saveas(gcf, fullfile(outDir, 'linear_kf_trajectory_matlab.png'));
    close(gcf);
    fprintf('Saved MATLAB plot: %s\n', fullfile(outDir, 'linear_kf_trajectory_matlab.png'));
end
