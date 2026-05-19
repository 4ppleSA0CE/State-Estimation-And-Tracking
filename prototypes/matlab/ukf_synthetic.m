% Radar UKF cross-check against prototypes/python/ukf_synthetic.py reference export.
% Scenario (x0_truth, Q, R, z) comes from Python EkfScenarioConfig — re-run Python first.

function ukf_synthetic()
    thisDir = fileparts(mfilename('fullpath'));
    refPath = fullfile(thisDir, '..', 'output', 'ukf_synthetic_ref.mat');
    if ~isfile(refPath)
        error('ukf_synthetic:MissingRef', ...
            'Reference file not found. Run prototypes/python/ukf_synthetic.py first.\n  Expected: %s', refPath);
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

    alpha = 1e-3;
    beta = 2.0;
    kappa = 0.0;
    if isfield(ref, 'ukf_alpha')
        alpha = ref.ukf_alpha;
        beta = ref.ukf_beta;
        kappa = ref.ukf_kappa;
    end

    x0 = initialStateFromRadar(z, dt);
    [x_est, P_hist, y_hist, s_hist] = runFilter(z, F, Q, R, x0, P0, alpha, beta, kappa);

    maxErr = max(abs(x_est(:) - x_est_py(:)));
    tol = 1e-9;
    fprintf('Max |x_mat - x_py|: %.3e (tol %.1e)\n', maxErr, tol);
    if maxErr > tol
        error('ukf_synthetic:ParityFailed', 'MATLAB and Python trajectories differ beyond tolerance.');
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

function [x_est, P_hist, y_hist, s_hist] = runFilter(z, F, Q, R, x0, P0, alpha, beta, kappa)
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
        [x, P] = ukfPredict(x, P, F, Q, alpha, beta, kappa);
        [x, P, y, S] = ukfUpdate(x, P, z(k, :).', R, alpha, beta, kappa);
        x_est(k, :) = x.';
        P_hist(:, :, k) = P;
        y_hist(k, :) = y.';
        s_hist(:, :, k) = S;
    end
end

function [x, P] = ukfPredict(x, P, F, Q, alpha, beta, kappa)
    stateDim = numel(x);
    [lam, wm, wc] = unscentedWeights(stateDim, alpha, beta, kappa);
    chi = sigmaPoints(x, P, lam);
    chiPred = F * chi;
    x = weightedMean(chiPred, wm);
    P = weightedCovariance(chiPred, x, wc, Q);
end

function [x, P, y, S] = ukfUpdate(x, P, z, R, alpha, beta, kappa)
    stateDim = numel(x);
    measDim = numel(z);
    [lam, wm, wc] = unscentedWeights(stateDim, alpha, beta, kappa);
    chi = sigmaPoints(x, P, lam);
    nSig = size(chi, 2);
    zSigmas = zeros(measDim, nSig);
    for i = 1:nSig
        zSigmas(:, i) = measurementModel(chi(:, i));
    end
    zPred = measurementMean(zSigmas, wm);
    y = z - zPred;
    y(2) = wrapAngle(y(2));
    Pzz = weightedCovariance(zSigmas, zPred, wc, R);
    diffX = chi - x;
    diffZ = zSigmas - zPred;
    Pxz = diffX * diag(wc) * diffZ.';
    K = Pxz / Pzz;
    x = x + K * y;
    P = P - K * Pzz * K.';
    P = 0.5 * (P + P.');
    P = P + 1e-12 * eye(stateDim);
    S = Pzz;
end

function [lam, wm, wc] = unscentedWeights(n, alpha, beta, kappa)
    lam = alpha^2 * (n + kappa) - n;
    wm = ones(2 * n + 1, 1) / (2 * (n + lam));
    wc = wm;
    wm(1) = lam / (n + lam);
    wc(1) = lam / (n + lam) + (1 - alpha^2 + beta);
end

function chi = sigmaPoints(x, P, lam)
    n = numel(x);
    x = x(:);
    scale = n + lam;
    [L, p] = chol(scale * P, 'lower');
    if p > 0
        P = P + 1e-9 * eye(n);
        L = chol(scale * P, 'lower');
    end
    chi = zeros(n, 2 * n + 1);
    chi(:, 1) = x;
    for i = 1:n
        chi(:, i + 1) = x + L(:, i);
        chi(:, n + i + 1) = x - L(:, i);
    end
end

function m = weightedMean(sigmas, wm)
    m = sigmas * wm;
end

function P = weightedCovariance(sigmas, meanVec, wc, noise)
    diff = sigmas - meanVec(:);
    P = diff * diag(wc) * diff.';
    if nargin >= 4 && ~isempty(noise)
        P = P + noise;
    end
end

function zPred = measurementMean(zSigmas, wm)
    rMean = sum(wm .* zSigmas(1, :).');
    c = sum(wm .* cos(zSigmas(2, :).'));
    s = sum(wm .* sin(zSigmas(2, :).'));
    zPred = [rMean; atan2(s, c)];
end

function z = measurementModel(x)
    px = x(1);
    py = x(2);
    r = hypot(px, py);
    z = [r; atan2(py, px)];
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
    legend('Ground truth', 'Meas. (Cart.)', 'UKF estimate', 'Radar', 'Location', 'best');

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

    saveas(gcf, fullfile(outDir, 'ukf_synthetic_summary_matlab.png'));
    close(gcf);
    fprintf('Saved MATLAB plot: %s\n', fullfile(outDir, 'ukf_synthetic_summary_matlab.png'));
end
