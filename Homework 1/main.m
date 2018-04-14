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

%rng('default')
data_0 = mvnrnd(m_0, C_0, N)';

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

data_1 = random(dist_1,N)';

% Plot the generated data
figure(1); clf
scatter(data_0(1,:), data_0(2,:),'+')
hold on
scatter(data_1(1,:), data_1(2,:),'+r')
legend('Class 0', 'Class 1')
title('200 samples generated from each class')

% Save distribution parameters to .mat-file
save params.mat m_0 m_1A m_1B C_0 C_1A C_1B pi_1A pi_1B

%% 2) Use MAP to classify samples and compute the decision boundary

data = [data_0, data_1];
% Classify samples according to maximum likelihood
t = classifyByMAP(data); % t is a vector containing the labels of all samples

% Define the ranges and resolution of the grid used for computing the
% decision boundary
u = linspace(-10, 10, 100);
v = linspace(-10, 10, 100);

zp = zeros(length(u), length(v));

% Evaluate zp = p_0 - p_1 over the grid
for i = 1:length(u)
    for j = 1:length(v)
        x(:,j) = [u(i) v(j)]';
    end
    whos x
    p_uv_0  = getProbability(x, 0);
    p_uv_1  = getProbability(x, 1);
    zp(:,i) = p_uv_0 - p_uv_1;
end

% Plot the decision boundary with the samples
figure(2);
scatter(data_0(1,:), data_0(2,:),'+b')
hold on
scatter(data_1(1,:), data_1(2,:),'+r')
hold on
contour(u,v,zp, [0, 0],'k', 'LineWidth', 2)
title('Decision boundary using the MAP rule')
legend('Class 0','Class 1','Decision boundary')

%% 3) Estimation of conditional probability of incorrect classification with MAP

N_test = 5000;
Test_X_0 = mvnrnd(m_0, C_0, N_test)';
Test_X_1 = random(dist_1,N_test)';

data = [Test_X_0, Test_X_1];
t = classifyByMAP(data);

% Check which samples were correctly classified
corr_0 = 0;
corr_1 = 0;
for i=1:size(t,2)
    if (i<=N_test && t(i) == 0)
        corr_0 = corr_0 + 1;
    elseif (i>N_test && t(i) == 1)
        corr_1 = corr_1 + 1;
    end 
end
p_incorr_0 = (N_test - corr_0)/N_test
p_incorr_1 = (N_test - corr_1)/N_test

% Plot the decision boundary with the test samples
figure(3);
scatter(Test_X_0(1,:), Test_X_0(2,:),'.b')
hold on
scatter(Test_X_1(1,:), Test_X_1(2,:),'.r')
hold on
contour(u,v,zp, [0, 0],'k', 'LineWidth', 2)
title('Classification of 5000 samples using the MAP rule')
legend('Class 0','Class 1','Decision boundary')


%% 4) Kernelized logistic regression

N_training_samples = 200;
X_0 = mvnrnd(m_0, C_0, N_training_samples/2);
X_1 = random(dist_1,N_training_samples/2);
l = 10^-1;
X = [X_0; X_1]';
K = -3*ones(200);
for i=1:size(X,2)
    for j=1:size(X,2)
        K(i,j) = exp(-norm(X(:,i)-X(:,j))^2/(2*l^2));
    end
end

% Newton iterations
a = zeros(N_training_samples,1);
t = [zeros(N_training_samples/2,1); ones(N_training_samples/2,1)];

lambda = 1;
for k=1:10
    y = 1./(1+exp(-a'*K))';
    R = diag(y.*(1-y));
    H = K*R*K+lambda*K;
    a0 = a;

    a = a - H\K*(y-t+lambda*a);
    
%     if (abs(a-a0)<0.01)
%         k
%         break
%     end 
end


%% 5) Plotting of data points and decision boundary with the kernel trick


% Decision boundary

% Define the ranges of the grid
u = linspace(-10, 10, 200);
v = linspace(-10, 10, 200);

% Initialize space for the values to be plotted
zk = zeros(length(u), length(v));

% Evaluate z = a'*K over the grid
X = [X_0; X_1]';
K_uv = -3*ones(200);

for i=1:size(X,2)
    for j=1:size(X,2)
        x_uv = [u(i) ; v(j)];
%         for m=1:size(X,2)
%             for n=1:size(X,2)
%                 K_uv(m,n) = exp(-norm(X(:,m)-x_uv)^2/(2*l^2));
%             end
%         end
        K_uv(i,j) = exp(-norm(X(:,i)-x_uv)^2/(2*l^2));
    end
end
for i = 1:length(u)
    zk(:,i) = a'*K_uv;
end

% Plot the generated data
figure(1);
scatter(X_0(:,1), X_0(:,2),'+')
hold on
scatter(X_1(:,1), X_1(:,2),'+r')
contour(u,v,zk, [0, 0],'k', 'LineWidth', 2)
title('Classification from ASD samples using the kernel trick')
legend('Class 0','Class 1','Decision boundary')

%% 6)
% Ez?

%% 7)

% Map the data to 3rd degree feature vectors, gather in matrix form
Phi = createFeatureMatrix([data_0 data_1]);
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
zp = zeros(length(u), length(v));

% Evaluate z = theta*x over the grid
for i = 1:length(u)
    for j = 1:length(v)
        phi_uv = createFeatureMatrix([u(i) v(j)]');
        zp(j,i) = w'*phi_uv;
    end
end

% Plot the generated data
figure(2);
scatter(data_0(1,:), data_0(2,:),'+')
hold on
scatter(data_1(1,:), data_1(2,:),'+r')
contour(u,v,zp, [0, 0],'k', 'LineWidth', 2)
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

Phi = createFeatureMatrix(MAP_test_samples_0, MAP_test_samples_1, 2);

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
contour(u,v,zp, [0, 0],'k', 'LineWidth', 2)
title('Simulated classification of 1000 samples using the MAP rule')
legend('Class 0','Class 1','Decision boundary')