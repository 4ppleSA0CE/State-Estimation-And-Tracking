% IMM (CV/CA/CT) cross-check against prototypes/python/imm_synthetic.py reference export.
% Scenario comes from Python ImmScenarioConfig — re-run Python first.

function imm_synthetic()
    thisDir = fileparts(mfilename('fullpath'));
    refPath = fullfile(thisDir, '..', 'output', 'imm_synthetic_ref.mat');
    if ~isfile(refPath)
        error('imm_synthetic:MissingRef', ...
            'Reference file not found. Run prototypes/python/imm_synthetic.py first.\n  Expected: %s', refPath);
    end

    ref = load(refPath);
    dt = ref.dt;
    R = ref.R;
    Pi = ref.Pi;
    z = ref.z;
    x_true = ref.x_true;
    x_imm_py = ref.x_imm_py;
    turnRate = ref.turn_rate_rad_s;
    cvDur = ref.cv_duration_s;
    ctDur = ref.ct_duration_s;

    sigmaPos = sqrt(R(1, 1));
    qAccel = 0.05;
    [x_imm, mu_hist] = runImm(z, dt, R, Pi, turnRate, sigmaPos, qAccel, cvDur, ctDur);

    maxErr = max(abs(x_imm(:) - x_imm_py(:)));
    tol = 1e-8;
    fprintf('Max |x_imm_mat - x_imm_py|: %.3e (tol %.1e)\n', maxErr, tol);
    if maxErr > tol
        error('imm_synthetic:ParityFailed', 'MATLAB and Python IMM trajectories differ.');
    end
    fprintf('Parity check: PASS\n');
    fprintf('Final mode probs: [%.3f, %.3f, %.3f]\n', mu_hist(end, :));

    plotSummary(z, x_true, x_imm, mu_hist, cvDur, ctDur, thisDir);
end

function [x_imm, mu_hist] = runImm(z, dt, R, Pi, turnRate, sigmaPos, qAccel, cvDur, ctDur)
    n = size(z, 1);
    stateDim = 4;
    numModes = 3;
    mu = ones(1, numModes) / numModes;
    mu_hist = zeros(n, numModes);

    p0 = diag([R(1,1), R(2,2), 10, 10]);
    x0 = initialStateFromMeas(z, dt);

    filt = initFilters(dt, R, turnRate, sigmaPos, qAccel);
    for j = 1:numModes
        filt{j} = setFilterState(filt{j}, x0, p0, j);
    end

    x_imm = zeros(n, stateDim);
    x_imm(1, :) = x0.';
    mu_hist(1, :) = mu;

    for k = 2:n
        states = cell(1, numModes);
        covs = cell(1, numModes);
        for j = 1:numModes
            [states{j}, covs{j}] = filterRefState(filt{j}, j);
        end
        [mixedX, mixedP, c] = mixImm(states, covs, mu, Pi);

        likelihoods = zeros(1, numModes);
        for j = 1:numModes
            filt{j} = setFilterState(filt{j}, mixedX{j}, mixedP{j}, j);
            filt{j} = predictFilter(filt{j}, j, dt, turnRate, sigmaPos, qAccel);
            [filt{j}, likelihoods(j)] = updateFilter(filt{j}, z(k, :).', R, j);
        end

        states = cell(1, numModes);
        covs = cell(1, numModes);
        for j = 1:numModes
            [states{j}, covs{j}] = filterRefState(filt{j}, j);
        end
        denom = sum(likelihoods .* c);
        if denom < 1e-300
            mu = ones(1, numModes) / numModes;
        else
            mu = (likelihoods .* c) / denom;
            mu = max(mu, 1e-12);
            mu = mu / sum(mu);
        end
        x_imm(k, :) = combinedState(states, mu).';
        mu_hist(k, :) = mu;
    end
end

function x0 = initialStateFromMeas(z, dt)
    v0 = (z(2, :) - z(1, :)) / dt;
    x0 = [z(1, 1); z(1, 2); v0(1); v0(2)];
end

function filt = initFilters(dt, R, turnRate, sigmaPos, qAccel)
    filt = cell(1, 3);
    filt{1} = struct('type', 'cv', 'dt', dt, 'R', R, 'sigmaPos', sigmaPos, 'qAccel', qAccel);
    filt{2} = struct('type', 'ca', 'dt', dt, 'R', R, 'sigmaPos', sigmaPos, 'qAccel', qAccel);
    filt{3} = struct('type', 'ct', 'dt', dt, 'R', R, 'turnRate', turnRate, 'qAccel', qAccel);
end

function filt = setFilterState(filt, xRef, pRef, modeIdx)
    switch filt.type
        case 'cv'
            filt.x = xRef(:);
            filt.P = pRef;
        case 'ca'
            x = zeros(6, 1);
            x(1:4) = xRef(:);
            P = diag([10, 10, 10, 10, 0.5, 0.5]);
            P(1:4, 1:4) = pRef;
            filt.x = x;
            filt.P = P;
        case 'ct'
            filt.x = xRef(:);
            filt.P = pRef;
    end
end

function [xRef, pRef] = filterRefState(filt, ~)
    switch filt.type
        case 'cv'
            xRef = filt.x;
            pRef = filt.P;
        case 'ca'
            xRef = filt.x(1:4);
            pRef = filt.P(1:4, 1:4);
        case 'ct'
            xRef = filt.x;
            pRef = filt.P;
    end
end

function filt = predictFilter(filt, modeIdx, dt, turnRate, sigmaPos, qAccel)
    switch filt.type
        case 'cv'
            [F, ~, Q, ~] = cvMatrices(dt, sigmaPos, qAccel);
            filt.x = F * filt.x;
            filt.P = F * filt.P * F.' + Q;
        case 'ca'
            [F, ~, Q, ~] = caMatrices(dt, sigmaPos, qAccel);
            filt.x = F * filt.x;
            filt.P = F * filt.P * F.' + Q;
        case 'ct'
            [filt.x, filt.P] = ukfCtPredict(filt.x, filt.P, dt, turnRate, qAccel);
    end
end

function [filt, likelihood] = updateFilter(filt, z, R, modeIdx)
    switch filt.type
        case 'cv'
            [~, H, ~, ~] = cvMatrices(filt.dt, filt.sigmaPos, filt.qAccel);
            [filt, likelihood] = kfUpdate(filt, z, H, R);
        case 'ca'
            [~, H, ~, ~] = caMatrices(filt.dt, filt.sigmaPos, filt.qAccel);
            [filt, likelihood] = kfUpdate(filt, z, H, R);
        case 'ct'
            [filt.x, filt.P, likelihood] = ukfCtUpdate(filt.x, filt.P, z, R, filt.dt, filt.qAccel);
    end
end

function [x, P] = ukfCtPredict(x, P, dt, turnRate, qAccel)
    alpha = 1e-3; beta = 2; kappa = 0;
    [lam, wm, wc] = unscentedWeights(4, alpha, beta, kappa);
    chi = sigmaPoints(x, P, lam);
    nSig = size(chi, 2);
    chiPred = zeros(4, nSig);
    for i = 1:nSig
        chiPred(:, i) = ctPropagate(chi(:, i), dt, turnRate);
    end
    x = weightedMean(chiPred, wm);
    g = [0.5*dt*dt; dt];
    q1 = qAccel * (g * g.');
    Q = zeros(4);
    Q([1,3],[1,3]) = q1;
    Q([2,4],[2,4]) = q1;
    P = weightedCovariance(chiPred, x, wc, Q);
end

function [x, P, likelihood] = ukfCtUpdate(x, P, z, R, dt, qAccel)
    alpha = 1e-3; beta = 2; kappa = 0;
    H = [1 0 0 0; 0 1 0 0];
    [lam, wm, wc] = unscentedWeights(4, alpha, beta, kappa);
    chi = sigmaPoints(x, P, lam);
    nSig = size(chi, 2);
    zSigmas = H * chi;
    zPred = weightedMean(zSigmas, wm);
    y = z - zPred;
    Pzz = weightedCovariance(zSigmas, zPred, wc, R);
    diffZ = zSigmas - zPred;
    Pxz = (chi - x) * diag(wc) * diffZ.';
    K = Pxz / Pzz;
    x = x + K * y;
    P = P - K * Pzz * K.';
    P = 0.5 * (P + P.') + 1e-12 * eye(4);
    likelihood = gaussianLikelihood(y, Pzz);
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

function [filt, likelihood] = kfUpdate(filt, z, H, R)
    x = filt.x;
    P = filt.P;
    hx = H * x;
    y = z - hx;
    S = H * P * H.' + R;
    K = P * H.' / S;
    filt.x = x + K * y;
    I = eye(numel(x));
    filt.P = (I - K * H) * P * (I - K * H).' + K * R * K.';
    likelihood = gaussianLikelihood(y, S);
end

function L = gaussianLikelihood(y, S)
    m = numel(y);
    L = exp(-0.5 * (m*log(2*pi) + log(max(det(S), 1e-30)) + y.' / S * y));
    L = max(L, 1e-300);
end

function [mixedX, mixedP, c] = mixImm(states, covs, mu, Pi)
    numModes = numel(states);
    c = (Pi.' * mu.').';
    c = max(c, 1e-12);
    mixedX = cell(1, numModes);
    mixedP = cell(1, numModes);
    for j = 1:numModes
        x0 = zeros(4, 1);
        for i = 1:numModes
            mu_ij = Pi(i, j) * mu(i) / c(j);
            x0 = x0 + mu_ij * states{i};
        end
        P0 = zeros(4);
        for i = 1:numModes
            mu_ij = Pi(i, j) * mu(i) / c(j);
            dx = states{i} - x0;
            P0 = P0 + mu_ij * (covs{i} + dx * dx.');
        end
        mixedX{j} = x0;
        mixedP{j} = P0;
    end
end

function x = combinedState(states, mu)
    x = zeros(4, 1);
    for j = 1:numel(states)
        x = x + mu(j) * states{j};
    end
end

function x = ctPropagate(x, dt, omega)
    px = x(1); py = x(2); vx = x(3); vy = x(4);
    if abs(omega) < 1e-8
        x = [px + vx*dt; py + vy*dt; vx; vy];
        return;
    end
    s = sin(omega*dt); c = cos(omega*dt);
    vx_n = c*vx - s*vy;
    vy_n = s*vx + c*vy;
    px_n = px + (vx*s + vy*(1-c))/omega;
    py_n = py + (vy*s - vx*(1-c))/omega;
    x = [px_n; py_n; vx_n; vy_n];
end

function [F, H, Q, R] = cvMatrices(dt, sigmaPos, qAccel)
    F = [1, 0, dt, 0; 0, 1, 0, dt; 0, 0, 1, 0; 0, 0, 0, 1];
    H = [1, 0, 0, 0; 0, 1, 0, 0];
    g = [0.5*dt*dt; dt];
    q1 = qAccel * (g * g.');
    Q = zeros(4);
    Q([1,3],[1,3]) = q1;
    Q([2,4],[2,4]) = q1;
    R = sigmaPos^2 * eye(2);
end

function [F, H, Q, R] = caMatrices(dt, sigmaPos, qAccel)
    f1 = [1, dt, 0.5*dt*dt; 0, 1, dt; 0, 0, 1];
    F = zeros(6);
    F([1,3,5],[1,3,5]) = f1;
    F([2,4,6],[2,4,6]) = f1;
    H = zeros(2, 6);
    H(1,1) = 1; H(2,2) = 1;
    g = [0.5*dt*dt; dt; 1];
    q1 = qAccel * (g * g.');
    Q = zeros(6);
    Q([1,3,5],[1,3,5]) = q1;
    Q([2,4,6],[2,4,6]) = q1;
    R = sigmaPos^2 * eye(2);
end

function plotSummary(z, x_true, x_imm, mu_hist, cvDur, ctDur, thisDir)
    outDir = fullfile(thisDir, '..', 'output');
    if ~exist(outDir, 'dir'); mkdir(outDir); end
    t = (0:size(z,1)-1).' * 0.1;
    figure('Visible', 'off');
    subplot(2, 2, 1);
    plot(x_true(:,1), x_true(:,2), 'k-'); hold on;
    plot(x_imm(:,1), x_imm(:,2), 'b-');
    axis equal; grid on; title('Trajectory');
    subplot(2, 2, 2);
    plot(t, mu_hist); grid on; title('Mode probabilities');
    subplot(2, 2, 3);
    err = sqrt(sum((x_true(:,1:2)-x_imm(:,1:2)).^2, 2));
    plot(t, err); grid on; title('IMM position error');
    saveas(gcf, fullfile(outDir, 'imm_synthetic_summary_matlab.png'));
    close(gcf);
end
