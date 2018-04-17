function [Phi] = createFeatureMatrix(X, N)

% X_0 = X(:,1:end/2);
% X_0 = X_0(:,1:N/2);
% X_1 = X(:,end/2+1:end);
% X_1 = X_1(:,1:N/2);

%X = [X_0 X_1];
x1 = X(1,:);
x2 = X(2,:);

Phi = [ ones(1,N);
        x1
        x2;
        x1.^2;
        x2.^2;
        x1.*x2;
        x1.^3;
        x1.^2.*x2;
        x1.*x2.^2;
        x2.^3     ];

end