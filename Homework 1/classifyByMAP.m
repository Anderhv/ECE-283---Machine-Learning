function [t] = classifyByMAP(X)

p_0 = getProbability(X, 0);
p_1 = getProbability(X, 1);

for i=1:size(X,2)
    if (log(p_0(i)) > log(p_1(i)))
        t(i) = 0;
    else
        t(i) = 1;
    end
end

end
