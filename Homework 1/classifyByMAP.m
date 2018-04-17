function [t_hat] = classifyByMAP(X)

N = length(X);

p_0 = getProbability(X, 0);
p_1 = getProbability(X, 1);
for i=1:length(X)
    if (log(p_0(i)) > log(p_1(i)))
        t_hat(i) = 0;
    else
        t_hat(i) = 1;
    end
end

% Check which samples were correctly classified
corr_0 = 0;
corr_1 = 0;
for i=1:N
    if (i<=N/2 && t_hat(i) == 0)
        corr_0 = corr_0 + 1;
    elseif (i>N/2 && t_hat(i) == 1)
        corr_1 = corr_1 + 1;
    end 
end
% t_hat denotes the estimated label, t denotes the actual label
fprintf('P(t_hat = 1 | t = 0) using MAP: %.3f\n',(N/2 - corr_0)/(N/2))
fprintf('P(t_hat = 0 | t = 1) using MAP: %.3f\n',(N/2 - corr_1)/(N/2))


end
