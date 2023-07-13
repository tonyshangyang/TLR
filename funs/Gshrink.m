function [x,objV] = Gshrink(x,rho,sX, isWeight,mode)

if isWeight == 1
%     C = 2*sqrt(2)*sqrt(sX(3)*sX(2));
    C = sqrt(sX(3)*sX(2));
end
if ~exist('mode','var')

    mode = 1;
end

X=reshape(x,sX);%%N*N*V
if mode == 1    %%%X是顺序张量，重构
    Y=X2Yi(X,3);
elseif mode == 3  %%X是一个矩阵，展开
    Y=shiftdim(X, 1);%%N*V*N
else
    Y = X;  %%%已经是高阶张量
end
% 

Yhat = fft(Y,[],3);%%傅里叶变换
% weight = C./sTrueV+eps;
% weight = 1;
% tau = rho*weight;
objV = 0;
if mode == 1
    n3 = sX(2);
elseif mode == 3
    n3 = sX(1);%%N
else
    n3 = sX(3);
end



    for i = 1:n3
        [uhat,shat,vhat] = svd(full(Yhat(:,:,i)),'econ');%%
        
            tau = rho;
            shat = max(shat - tau,0);   %%硬阈值化
               
        
        objV = objV + sum(shat(:));
        Yhat(:,:,i) = uhat*shat*vhat';

    end
  

Y = ifft(Yhat,[],3);%%逆傅里叶变换
if mode == 1
    X = Yi2X(Y,3);
elseif mode == 3
    X = shiftdim(Y, 2);%%N*N*V
else
    X = Y;
end

x = X(:);%%向量化

end
 