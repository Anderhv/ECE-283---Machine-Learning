%% Homework 1 - Classification using Logistic Regression
% Anders Haver Vagle, April 2018
% In Homework group with: Sondre Kongsgaard, Morten Lie, Brage Saether,
% Franky Meng, Yang Zhao, Yulin OU

%% 1) Generate 2D synthetic data
N = 200;

% Class 0
m_0 = [0, 0]';
lambda_01 = 2;
lambda_02 = 1;
theta_0 = 0;
u_01 = [cos(theta_0), sin(theta_0)]';
u_02 = [-sin(theta_0), cos(theta_0)]';
C_0 = lambda_01*(u_01*u_01') + lambda_02*(u_02*u_02');

rng(888)
X_0 = mvnrnd(m_0, C_0, N)';

% Class 1
% Component A
m_1A = [-2, 1]';
lambda_1A1 = 2;
lambda_1A2 = 1/4;
theta_1A = -3*pi/4;
u_1A1 = [cos(theta_1A), sin(theta_1A)]';
u_1A2 = [-sin(theta_1A), cos(theta_1A)]';
C_1A = lambda_1A1*(u_1A1*u_1A1') + lambda_1A2*(u_1A2*u_1A2');
pi_1A = 1/3;
% Component B
m_1B = [3, 2]';
lambda_1B1 = 3;
lambda_1B2 = 1;
theta_1B = pi/4;
u_1B1 = [cos(theta_1B), sin(theta_1B)]';
u_1B2 = [-sin(theta_1B), cos(theta_1B)]';
C_1B = lambda_1B1*(u_1B1*u_1B1') + lambda_1B2*(u_1B2*u_1B2');
pi_1B = 2/3;
% Mixture
dist_1 = gmdistribution([m_1A'; m_1B'], cat(3,C_1A,C_1B),[pi_1A, pi_1B]);

X_1 = random(dist_1,N)';

% Plot the generated data
figure(20); clf
scatter(X_0(1,:), X_0(2,:),'.b')
hold on
scatter(X_1(1,:), X_1(2,:),'.r')
legend('Class 0', 'Class 1','Location','SouthEast')
title('200 samples generated from each class')

% Save distribution parameters to .mat-file
save params.mat m_0 m_1A m_1B C_0 C_1A C_1B pi_1A pi_1B

%% 2) Use MAP to classify samples and compute the decision boundary

% Classify samples according to maximum likelihood
t = classifyByMAP([X_0 X_1]); % t is a vector containing the labels of all samples

% Define the ranges and resolution of the grid used for computing the
% decision boundary
res = 150;
u = linspace(-8, 8, res);
v = linspace(-8, 8, res);
z_MAP = zeros(res, res);

% Evaluate zp = p_0 - p_1 over the grid
for i = 1:length(u)
    for j = 1:length(v)
        x(:,j) = [u(i) v(j)]';
    end
    p_uv_0  = getProbability(x, 0);
    p_uv_1  = getProbability(x, 1);
    z_MAP(:,i) = p_uv_0 - p_uv_1;
end

% Plot the decision boundary with the samples
figure(21);
scatter(X_0(1,:), X_0(2,:),'.b')
hold on
scatter(X_1(1,:), X_1(2,:),'.r')
hold on
contour(u,v,z_MAP, [0, 0],'k', 'LineWidth', 2)
title('Decision boundary using the MAP rule')
legend('Class 0','Class 1','Decision boundary','Location', 'SouthEast')

%% 3) Estimation of conditional probability of incorrect classification with MAP

% Create a set of training samples for future tasks
N_training_samples = 1600;
rng(888)
X_0 = mvnrnd(m_0, C_0, N_training_samples/2)';
X_1 = random(dist_1,N_training_samples/2)';
X = [X_0 X_1];

t_hat = classifyByMAP(X);

% Plot the decision boundary with the test samples
figure(22); clf;
scatter(X_0(1,:), X_0(2,:),'.b')
hold on
scatter(X_1(1,:), X_1(2,:),'.r')
hold on
contour(u,v,z_MAP, [0, 0],'k', 'LineWidth', 2)
title('Classification of 1600 samples using the MAP rule')
legend('Class 0','Class 1','Decision boundary','Location', 'SouthEast')


%% 4) 5) 6) with: Kernelized Logistic Regression

l = 0.5;
lambda = 10;

a_20   = KLR(X, 20,  l, lambda);
a_50   = KLR(X, 50,  l, lambda);
a_100  = KLR(X, 100, l, lambda);
a_200  = KLR(X, 200, l, lambda);
a_400  = KLR(X, 400, l, lambda);
a_800  = KLR(X, 800, l, lambda);

% Evaluate z = a'*K, calculate probabilities and plot decision boundary
[z_20 ] = classifyByKLR(X, 20,  a_20,  l); % N = 20
[z_50 ] = classifyByKLR(X, 50,  a_50,  l); % N = 50
[z_100] = classifyByKLR(X, 100, a_100, l); % N = 100
[z_200] = classifyByKLR(X, 200, a_200, l); % N = 200
[z_400] = classifyByKLR(X, 400, a_400, l); % N = 400


%% 7) = 4) 5) 6) with: Non-kernelized Logistic Regression

[Phi_20,  w_20]  = NKLR(X, 20);
[Phi_50,  w_50]  = NKLR(X, 50);
[Phi_100, w_100] = NKLR(X, 100);
[Phi_200, w_200] = NKLR(X, 200); 
[Phi_400, w_400] = NKLR(X, 400);
[Phi_800, w_800] = NKLR(X, 800);
[Phi_1600, w_1600] = NKLR(X, 1600);

% Evaluate z = w'*Phi, calculate probabilities and plot decision boundary
z_100 = classifyByNKLR(X, 100, Phi_100, w_100);
z_200 = classifyByNKLR(X, 200, Phi_200, w_200);
z_400 = classifyByNKLR(X, 400, Phi_400, w_400);
z_800 = classifyByNKLR(X, 800, Phi_800, w_800);
z_1600 = classifyByNKLR(X, 1600, Phi_1600, w_1600);

%% Conclusion
% The MAP rule gives a decent indicator for what we could expect as the
% best possible probabilities for correct classification of samples.
% None of these models should thus be expected to have less than 15 %
% chance of false classification. We see that the KLR method does not
% converge for the chosen l and lambda with more than 400 points, while
% NKLR does not converge with less than 100 points. The NKLR thus seems to
% be better suited for larger sample sizes, and also runs faster than KLR. 

%% Functions
