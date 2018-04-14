function [Phi] = createFeatureMatrix(X)
N = length(X(1,:));
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