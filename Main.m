function [S,W,DD,history,Obj_all,objV]=Main(X,lambda,lambda2,dim_Reduce,k)

num_view=size(X,2);
nFea=zeros(1,num_view);
num_X=size(X{1,1},2);  %样本个数

R=cell(1,num_view);
G=cell(1,num_view);

for v=1:num_view
    R{v} = zeros(num_X,num_X);  %Lagrange multiplier
    G{v} = zeros(num_X,num_X);   % variable S
    nFea(1,v)=size(X{1,v},1);    %特征的维度
end

Isconverg = zeros(1,num_view);epson = 1e-5;
iter = 0;
rho=0.3*num_X; max_rho = 10e12; pho_rho =1.3;

% S = InitializeS(X);  图矩阵
S=cell(1,num_view);
distX=cell(1,num_view);
distX1=cell(1,num_view); %把距离排序的结果
idx=cell(1,num_view);  %排序后的索引
for v=1:num_view
    
    distX{v} = L2_distance_1(X{v},X{v});
    [distX1{v}, idx{v}] = sort(distX{v},2); %对每行中的元素进行排序
											%%将排序后的矩阵赋值给distX1{v}，同时将对应的索引赋值给idx{v}。
    S{1,v} = zeros(num_X);
    for i = 1:num_X
        di = distX1{v}(i,2:k+2);  %选第二列到k+2列
        id = idx{v}(i,2:k+2);
        S{1,v}(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
    end
    %     S{1,v}=(S{1,v}+S{1,v}');
end

% Initialize D
dd=cell(1,num_view);
DD=cell(1,num_view);  %O
for v=1:num_view
    dd{v}=ones(nFea(v),1);
    DD{v} = spdiags(dd{v},0,nFea(v),nFea(v));  %通过获取dd{v}的列并沿对角线放置它们，来创建一个nFea(v)*nFea(v)的稀疏矩阵
    DD{v}=full(DD{v});  %将稀疏矩阵转换为满存储
end

Obj_all=[];
while(sum(Isconverg) == 0)  %是否收敛
    fprintf('----processing iter %d--------\n', iter+1);
    
    
    %%% W D  
    tS=cell(1,num_view);
    Ls=cell(1,num_view);
    W=cell(1,num_view);
    
    H=eye(num_X)-(ones(num_X,1)*ones(1,num_X))./num_X;  %eye单位矩阵
    for v=1:num_view
        
        tS{v} = S{v}; %%%%%%%%%%%%%%%%%%%%%
        A0=[];
        A0=tS{v};
 
        Ls{v} = full(diag(sum(A0,2))-A0);  %diag创建对角矩阵，拉普拉斯乘子 L
        
        temp_lambda=lambda;
        temp_A=X{1,v}*Ls{v}*X{1,v}' + temp_lambda*DD{v}; %P
        temp_A=max(temp_A,temp_A');

        if dim_Reduce < nFea(v)
            [W{v}, eigvalue] = eigs(temp_A,dim_Reduce,'sm'); %返回dim_Reduce个模最小的特征值
															 %%函数返回值中的W是一个矩阵，每列是temp_A的特征向量，对应着eigvalue中的特征值
        else
            dim_Reduce = nFea(v);
            [W{v}, eigvalue] = eigs(temp_A,dim_Reduce,'sm');
        end      
        
        dd{v} = sqrt(diag(W{v}*W{v}'));
        epsilon = 0.1*max(dd{v});
        DD{v} =  (1/2)*diag(1./(dd{v}+epsilon));

    end
    
    
    %% Step 3: Fixed alpha and W, Update S
    if iter>0
        
        beta=cell(1,num_view);
        Z=cell(1,num_view);
        M=cell(1,num_view);
        WX=cell(1,num_view);
        for v=1:num_view
            WX{v} = X{v}'*W{v};
            distx = [];
            distx = L2_distance_1(WX{v}',WX{v}');
            
            distx=distx-diag(diag(distx));   %distx 矩阵的对角线元素设置为零，并保持其他元素不变
            beta{v}=distx;
            Z{v}=G{v}-R{v}./rho;

            M{v}=Z{v}-beta{v}./rho;%%不是真m

            
            S{1,v} = zeros(num_X);
            for i=1:num_X

                temp_i=1:num_X;
                
                    S{1,v}(i,temp_i~=i) =opt_S(M{v}(i,temp_i~=i));  %更新了矩阵S{1,v}除对角线部分的值
            end
        end
        
    end
    
    %% update G
    S_tensor = cat(3, S{:,:});%%n*n*l %沿着第三个维度将矩阵连接
    R_tensor = cat(3, R{:,:});
    temp_S = S_tensor(:);%%%向量化
    temp_R = R_tensor(:);%%%要先把他们向量化再进行加减
    
    sX = [num_X, num_X, num_view];
    %twist-version
    [g, objV] = Gshrink(temp_S + 1/rho*temp_R,(num_X*lambda2)/rho,sX,0,3)   ; %%%%%%%%
    
    G_tensor = reshape(g, sX); %按照sX重构张量
    
    %5 update R
    temp_R = temp_R + rho*(temp_S - g);
    
    %record the iteration information
    history.objval(iter+1)   =  objV;
    
    %%
    
    temp_objall=0;
    alpha_g = zeros(num_view,1);
    for v=1:num_view
        temp_objall= temp_objall+sum(dd{1,v});
        WX{v} = X{v}'*W{v};
        alpha_g(v) = trace(WX{v}'*Ls{v}*WX{v});   %求迹
    end
    Obj_all(iter+1)=sum(alpha_g)+lambda2*objV+lambda*temp_objall;
    
    
    %% coverge condition
    
    
    Isconverg = ones(1,num_view);
    
    
    history.S_G(iter+1)=0;
    for v=1:num_view
        
        
        G{v} = G_tensor(:,:,v);%%%%拆开
        R_tensor = reshape(temp_R , sX);%%%%
        R{v} = R_tensor(:,:,v);%%%%
        history.norm_S_G = norm(S{v}-G{v},inf); %%矩阵的最大绝对行之和
        
        history.S_G(iter+1)=history.S_G(iter+1)+history.norm_S_G;
        if (abs(history.norm_S_G)>epson)
            
            %             fprintf('norm_S_G %7.10f    \n', history.norm_S_G);
            fprintf('----norm_S_G  %5.5f  \n', history.norm_S_G);
            
            Isconverg(v) = 0;
            
        end
    end
    history.S_G(iter+1)=history.S_G(iter+1)/num_view; 
    
    
    if (iter>200)   %%Isconverg为1就停止，为0就继续迭代
        Isconverg  = ones(1,num_view);
    end
    iter = iter + 1;
    rho = min(rho*pho_rho, max_rho);
    
    
end
end