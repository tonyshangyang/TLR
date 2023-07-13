function [x ft] = opt_S(v, k)

%
%% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=1
%

if nargin < 2
    k = 1;
end;

ft=1;
n = length(v);

v0 = v-mean(v) + k/n;%%真m
%vmax = max(v0);
vmin = min(v0);%%s的值一定要是正的
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10%%更新lambda
        v1 = v0 - lambda_m;%%更新s
        posidx = v1>0;
        npos = sum(posidx);
        g = -npos;
        f = sum(v1(posidx)) - k;%%s中负的都当作0
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(v1,0);
            break;
        end;
    end;
    x = max(v1,0);

else
    x = v0;
end;