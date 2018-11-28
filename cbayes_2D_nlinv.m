%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the mesh in the parameter domain to evaluate the model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(12345)

Ns = 10000;

xmin = 0.79;
xmax = 0.99;
ymin = 1-4.5*sqrt(0.1);
ymax = 1+4.5*sqrt(0.1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prior distribution on parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

spts = zeros(Ns,2);

vol = (xmax-xmin)*(ymax-ymin);

% Uniform prior
spts(:,1) = xmin + (xmax-xmin)*rand(Ns,1);
spts(:,2) = ymin + (ymax-ymin)*rand(Ns,1);
prior_param_dist = @(x,y) 1/vol +0*x+0*y;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate distribution on data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

stdev = 0.025;
qbar = 0.3;
data_dist = @(q) 1/stdev/sqrt(2*pi)*exp(-(q-qbar).^2 / 2 / stdev^2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluate model at new points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

evals = eval2Dmodel(spts);

% Just take the second component as the QoI
qvals = evals(:,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computed the Posterior distribution on parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

prior_vals = prior_param_dist(spts(:,1),spts(:,2));
data_vals = data_dist(qvals);
pfprior_vals = km_kdeND(qvals,[],qvals);

ratio_vals = data_vals./pfprior_vals;

post_vals = prior_vals.*ratio_vals;

Integral_of_posterior = mean(ratio_vals)

% Avoid nans in log (does not affect anything else)
I = find(abs(ratio_vals)<1e-10);
ratio_vals(I) = 1.0e-10;
KLdiv_prior_to_posterior = mean(ratio_vals.*log(ratio_vals))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the posterior
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
scatter(spts(:,1),spts(:,2),20,post_vals,'filled');
title('Posterior Density')
colorbar
colormap('hot')
xlim([xmin xmax])
ylim([ymin ymax])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Samples from the posterior (basic rejection sampling)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

check = rand(Ns,1);
ratio_vals = ratio_vals./max(ratio_vals); % post/prior = data/pfprior = ratio
I = find(ratio_vals >= check);
spts_keep = spts(I,:);
qvals_keep = qvals(I,:);

merr = (0.3 - mean(qvals_keep))
sderr = (0.025 - sqrt(var(qvals_keep)))

figure
scatter(spts_keep(:,1),spts_keep(:,2),'filled');
title('Samples Generated from the Posterior')
