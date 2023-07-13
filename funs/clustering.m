function [result1]=clustering(S, cls_num, gt)
[C] = SpectralClustering(S,cls_num);
[~, nmi , ~] = compute_nmi(gt,C);
ACC = Accuracy(C,double(gt));
[f,~,~] = compute_f(gt,C);
[~,RI,~,~]=RandIndex(gt,C);
result1=[nmi,ACC,f,RI];
% result1 = ClusteringMeasure(gt, C);
end
