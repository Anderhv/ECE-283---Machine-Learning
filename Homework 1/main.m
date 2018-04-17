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

rng(100)
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
figure(1); clf
scatter(X_0(1,:), X_0(2,:),'.b')
hold on
scatter(X_1(1,:), X_1(2,:),'.r')
legend('Class 0', 'Class 1')
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
figure(2);
scatter(X_0(1,:), X_0(2,:),'.b')
hold on
scatter(X_1(1,:), X_1(2,:),'.r')
hold on
contour(u,v,z_MAP, [0, 0],'k', 'LineWidth', 2)
title('Decision boundary using the MAP rule')
legend('Class 0','Class 1','Decision boundary')

%% 3) Estimation of conditional probability of incorrect classification with MAP

% Create a set of training samples for future tasks
N_training_samples = 1600;
rng('default')
X_0 = mvnrnd(m_0, C_0, N_training_samples/2)';
X_1 = random(dist_1,N_training_samples/2)';
X = [X_0 X_1];

t_hat = classifyByMAP(X);

% Plot the decision boundary with the test samples
figure(3); clf;
scatter(X_0(1,:), X_0(2,:),'.b')
hold on
scatter(X_1(1,:), X_1(2,:),'.r')
hold on
contour(u,v,z_MAP, [0, 0],'k', 'LineWidth', 2)
title('Classification of 1600 samples using the MAP rule')
legend('Class 0','Class 1','Decision boundary')


%% 4) 5) 6) with: Kernelized Logistic Regression

l = 0.5;
lambda = 1000;

%a_20   = KLR(X, 20,  l, lambda);
%a_50   = KLR(X, 50,  l, lambda);
a_100  = KLR(X, 100, l, lambda);
a_200  = KLR(X, 200, l, lambda);
a_400  = KLR(X, 400, l, lambda);
%a_800  = KLR(X, 800, l, lambda);
%a_1600 = KLR(X_1600);

% Evaluate z = a'*K, calculate probabilities and plot decision boundary
%[z_20 ] = classifyByKLR(X, 20,  a_20,  l); % N = 20
%[z_50 ] = classifyByKLR(X, 50,  a_50,  l); % N = 50
[z_100] = classifyByKLR(X, 100, a_100, l); % N = 100
[z_200] = classifyByKLR(X, 200, a_200, l); % N = 200
[z_400] = classifyByKLR(X, 400, a_400, l); % N = 400
%[z_800] = classifyByKLR(X, 800, a_800, l); % N = 800


%% 7) = 4) 5) 6) with: Non-kernelized Logistic Regression

% Map the data to 3rd degree feature vectors, gather in matrix form
Phi = createFeatureMatrix(X);
w = zeros(size(Phi(:,1)));
t = [zeros(1,N) ones(1,N)];

for k=1:50
    y = 1./(1+exp(-w'*Phi));
    R = diag(y.*(1-y));
    w0 = w;
    w = w - (Phi*R*Phi')\Phi*(y-t)';
    if (abs(w-w0)<0.001)
        k
        break
    end 
end

% Decision boundary
% Define the ranges of the grid
u = linspace(-10, 10, 200);
v = linspace(-10, 10, 200);

% Initialize space for the values to be plotted
z_MAP = zeros(length(u), length(v));

% Evaluate z = theta*x over the grid
for i = 1:length(u)
    for j = 1:length(v)
        phi_uv = createFeatureMatrix([u(i) v(j)]');
        z_MAP(j,i) = w'*phi_uv;
    end
end

% Plot the generated data
figure(2);
scatter(data_0(1,:), data_0(2,:),'+')
hold on
scatter(data_1(1,:), data_1(2,:),'+r')
contour(u,v,z_MAP, [0, 0],'k', 'LineWidth', 2)
title('Classification from 200 samples using the MAP rule')
legend('Class 0','Class 1','Decision boundary')

% Classify generated samples
class_0_correct = 0;
class_1_correct = 0;
for i = 1:2*N
    z = w'*Phi(:,i);
    if (z < 0 && i <= N)
        % Sample from Class 0 classified as Class 0
        class_0_correct = class_0_correct + 1;
    end
    if (z > 0 && i > N)
        % Sample from Class 1 classified as Class 1
        class_1_correct = class_1_correct + 1;
    end     
end
correct_classification_percentage = ((class_0_correct + class_1_correct) / (2*N))*100


%% 3) 7test? Estimation of conditional probability of incorrect classification with MAP
N_test = 5000;
MAP_test_samples_0 = mvnrnd(m_0, C_0, N_test);
MAP_test_samples_1 = random(dist_1,N_test);

Phi = createFeatureMatrix([MAP_test_samples_0 MAP_test_samples_1]);

class_0_correct = 0;
class_1_correct = 0;
for i = 1:2*N_test
    z = w'*Phi(:,i);
    if (z < 0 && i <= N_test)
        % Sample from Class 0 classified as Class 0
        class_0_correct = class_0_correct + 1;
    end
    if (z > 0 && i > N_test)
        % Sample from Class 1 classified as Class 1
        class_1_correct = class_1_correct + 1
    end     
end
prob_error_class_0 = (N_test - class_0_correct) / N_test
prob_error_class_1 = (N_test - class_1_correct) / N_test


% Plot the generated data
figure(1); clf
scatter(MAP_test_samples_0(:,1), MAP_test_samples_0(:,2),'.')
hold on
scatter(MAP_test_samples_1(:,1), MAP_test_samples_1(:,2),'.r')
contour(u,v,z_MAP, [0, 0],'k', 'LineWidth', 2)
title('Simulated classification of 1000 samples using the MAP rule')
legend('Class 0','Class 1','Decision boundary')
