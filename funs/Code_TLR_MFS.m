clc
clear
close all


load bbcsportdata

%% Initialization

k=5;
lambda=10;
lambda2=1;

% dim_Reduce=max(gt);
dim_Reduce=max(gt);

num_view=size(X,2);
nFea=zeros(1,num_view);
num_X=size(X{1,1},2);


 temp_std=cell(1,num_view);
for v=1:num_view
    temp_std{v}=std(X{1,v},0,2);

    X{1,v}(temp_std{v}==0,:)=[];
    
end


X_Multi=[];
X_Multi2=[];
nFea=zeros(1,num_view);
 temp_std=cell(1,num_view);
for v=1:num_view
    
      X_Multi=[X_Multi;X{1,v}];
     temp_std{v}=std(X{1,v},0,2);
    for i=1:size(X{1,v},1)

           meanvalue_fea=mean(X{1,v}(i,:));

          X{1,v}(i,:)=(X{1,v}(i,:)-meanvalue_fea)/temp_std{v}(i,:);
   
    end
          X_Multi2=[X_Multi2;X{1,v}];  
               nFea(v)=size(X{v},1);
         
end


%% Main

[S,W,DD,history,Obj_all]=Main(X,lambda,lambda2,dim_Reduce,k);


score=[];
for v=1:num_view
    temp_score = sqrt(diag(W{v}*W{v}'));
    score=[score ;temp_score];
    
end


[~, mrrfs_f_idx] = sort(score,'descend');
Fea_fs=X_Multi(mrrfs_f_idx,:);


temp_fea_num=0.02:0.02:0.12;
ACC_fs=zeros(6,50);
NMI_fs=zeros(6,50);

for i=1:6
    
    d_fea=size(X_Multi,1);
    
    fea_num=ceil(temp_fea_num(i)*d_fea);
    
    fea_fs=Fea_fs(1:fea_num,:);
    
    MAXiter = 500; % Maximum of iterations for KMeans
    REPlic = 20; % Number of replications for KMeans
    class_num=max(gt);
    idx=[];
    result=[];
    
    
    parfor ii=1:50
        idx = kmeans(fea_fs',class_num,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
        result = ClusteringMeasure(gt, idx);
        
        ACC_fs(i,ii)=result(1,1);
        NMI_fs(i,ii)=result(1,2);
    end
end

MeanACC_fs_Multi=mean(ACC_fs,2);
MeanNMI_fs_Multi=mean(NMI_fs,2);

