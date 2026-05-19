% Radar EKF cross-check against prototypes/python/ekf_synthetic.py reference export.
% Scenario (x0_truth, Q, R, z) comes from Python EkfScenarioConfig — re-run Python first.

function ekf_synthetic()
    thisDir = fileparts(mfilename('fullpath'));
    refPath = fullfile(thisDir, '..', 'output', 'ekf_synthetic_ref.mat');
    if ~isfile(refPath)
        error('ekf_synthetic:MissingRef', ...
            'Reference file not found. Run prototypes/python/ekf_synthetic.py first.\n  Expected: %s', refPath);
    end

    ref = load(refPath);
    dt = ref.dt;
    F = ref.F;
    Q = ref.Q;
    R = ref.R;
    z = ref.z;
    x_true = ref.x_true;
    x_est_py = ref.x_est_py;
    P0 = ref.P0;
    fprintf('Truth x0 (from ref): [%.2f, %.2f, %.2f, %.2f]\n', x_true(1, :));

    x0 = initialStateFromRadar(z, dt);
    [x_est, P_hist, y_hist, s_hist] = runFilter(z, F, Q, R, x0, P0, dt);

    maxErr = max(abs(x_est(:) - x_est_py(:)));
    tol = 1e-9;
    fprintf('Max |x_mat - x_py|: %.3e (tol %.1e)\n', maxErr, tol);
    if maxErr > tol
        error('ekf_synthetic:ParityFailed', 'MATLAB and Python trajectories differ beyond tolerance.');
    end
    fprintf('Parity check: PASS\n');

    nis = nisPerStep(y_hist, s_hist);
    fprintf('Mean NIS (post step 1): %.3f\n', mean(nis(2:end)));

    plotSummary(z, x_true, x_est, y_hist, nis, thisDir);
end

function x0 = initialStateFromRadar(z, dt)
    p0 = cartesianFromRadar(z(1, :));
    p1 = cartesianFromRadar(z(2, :));
    v0 = (p1 - p0) / dt;
    x0 = [p0(1); p0(2); v0(1); v0(2)];
end

function xy = cartesianFromRadar(z)
    xy = [z(1) * cos(z(2)), z(1) * sin(z(2))];
end

function [x_est, P_hist, y_hist, s_hist] = runFilter(z, F, Q, R, x0, P0, dt)
    n = size(z, 1);
    stateDim = 4;
    measDim = 2;
    x = x0(:);
    P = P0;
    x_est = zeros(n, stateDim);
    P_hist = zeros(stateDim, stateDim, n);
    y_hist = zeros(n, measDim);
    s_hist = zeros(measDim, measDim, n);

    x_est(1, :) = x.';
    P_hist(:, :, 1) = P;
    y_hist(1, :) = 0;
    s_hist(:, :, 1) = eye(measDim);

    for k = 2:n
        [x, P] = ekfPredict(x, P, F, Q);
        [x, P, y, S] = ekfUpdate(x, P, z(k, :).', R);
        x_est(k, :) = x.';
        P_hist(:, :, k) = P;
        y_hist(k, :) = y.';
        s_hist(:, :, k) = S;
    end
end

function [x, P] = ekfPredict(x, P, F, Q)
    x = F * x;
    P = F * P * F.' + Q;
end

function [x, P, y, S] = ekfUpdate(x, P, z, R)
    hx = measurementModel(x);
    y = z - hx;
    y(2) = wrapAngle(y(2));
    H = measurementJacobian(x);
    S = H * P * H.' + R;
    K = P * H.' / S;
    x = x + K * y;
    I = eye(numel(x));
    P = (I - K * H) * P * (I - K * H).' + K * R * K.';
end

function z = measurementModel(x)
    px = x(1);
    py = x(2);
    r = hypot(px, py);
    z = [r; atan2(py, px)];
end

function H = measurementJacobian(x)
    px = x(1);
    py = x(2);
    r2 = px * px + py * py;
    r = max(sqrt(r2), 1e-6);
    r2 = max(r2, r * r);
    H = [px / r, py / r, 0, 0;
         -py / r2, px / r2, 0, 0];
end

function a = wrapAngle(a)
    a = mod(a + pi, 2 * pi) - pi;
end

function nis = nisPerStep(y_hist, s_hist)
    n = size(y_hist, 1);
    nis = zeros(n, 1);
    for k = 2:n
        y = y_hist(k, :).';
        S = s_hist(:, :, k);
        nis(k) = y.' * (S \ y);
    end
end

function plotSummary(z, x_true, x_est, y_hist, nis, thisDir)
    outDir = fullfile(thisDir, '..', 'output');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    n = size(z, 1);
    meas_xy = zeros(n, 2);
    for k = 1:n
        meas_xy(k, :) = cartesianFromRadar(z(k, :));
    end

    figure('Visible', 'off');
    subplot(2, 2, 1);
    hold on;
    plot(x_true(:, 1), x_true(:, 2), 'k-', 'LineWidth', 1.5);
    scatter(meas_xy(:, 1), meas_xy(:, 2), 8, 'filled', 'MarkerFaceAlpha', 0.35);
    plot(x_est(:, 1), x_est(:, 2), 'b-', 'LineWidth', 1.5);
    plot(0, 0, 'r^', 'MarkerSize', 8);
    axis equal;
    grid on;
    title('Trajectory');
    legend('Ground truth', 'Meas. (Cart.)', 'EKF estimate', 'Radar', 'Location', 'best');

    subplot(2, 2, 2);
    posErr = sqrt(sum((x_true(:, 1:2) - x_est(:, 1:2)).^2, 2));
    plot(posErr, 'b-');
    grid on;
    title('Position error');
    ylabel('[m]');

    subplot(2, 2, 3);
    plot(y_hist(2:end, 1));
    hold on;
    plot(y_hist(2:end, 2));
    grid on;
    title('Innovations');
    legend('range', 'bearing');

    subplot(2, 2, 4);
    plot(nis(2:end), 'b-');
    hold on;
    yline(2, 'k--');
    grid on;
    title('NIS');
    ylabel('NIS');

    saveas(gcf, fullfile(outDir, 'ekf_synthetic_summary_matlab.png'));
    close(gcf);
    fprintf('Saved MATLAB plot: %s\n', fullfile(outDir, 'ekf_synthetic_summary_matlab.png'));
end
