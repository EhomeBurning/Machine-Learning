% This code computes the false positive and accuracy for 1D, 2D, and 3D
% cases using KDD network anomaly dataset. 

%==========================================================================
%============================ Load Data ===================================
%==========================================================================
clear
close all
load http.mat
label   = y;
rawdata = X;

p       = 0.05; % Assume 5% outliers in obeservations
verbose = 1;    % Show text
m       = 2000; % Number of training samples

%==========================================================================
%======================= Generate 1D, 2D, 3D data =========================
%==========================================================================
coeff = pca(X);
meanX = mean(X);
X_1d = (X-meanX) * coeff(:,1)+meanX(1);
X_2d = (X-meanX) * coeff(:,1:2)+meanX(1:2);
X_3d = X;


NormalData_1d = X_1d(label==0,:);  % 0: inliers, 1: outliers
NormalData_2d = X_2d(label==0,:);
NormalData_3d = X_3d(label==0,:);
n = size(NormalData_1d,1);

randPermutation = randsample(n,m); 

X_train_1d = NormalData_1d(randPermutation',:);
X_train_2d = NormalData_2d(randPermutation',:);
X_train_3d = NormalData_3d(randPermutation',:);

%%------------------------ Train One-class SVM ----------------------------

SVMModel_1d = fitcsvm(X_train_1d,ones(m,1),'KernelScale','auto',...
    'Standardize',true,'OutlierFraction',p);
SVMModel_2d = fitcsvm(X_train_2d,ones(m,1),'KernelScale','auto',...
    'Standardize',true,'OutlierFraction',p);
SVMModel_3d = fitcsvm(X_train_3d,ones(m,1),'KernelScale','auto',...
    'Standardize',true,'OutlierFraction',p);

[~,score_1d] = predict(SVMModel_1d,X_1d);
[~,score_2d] = predict(SVMModel_2d,X_2d);
[~,score_3d] = predict(SVMModel_3d,X_3d);

%----------------------- Compute false positive ---------------------------
%--------------------------- Compute accuracy -----------------------------
FPost_1d = nnz(score_1d(label==0)<0)/n;
Accuracy_1d = nnz(score_1d(label == 1)<0)/nnz(label == 1);

FPost_2d = nnz(score_2d(label==0)<0)/n;
Accuracy_2d = nnz(score_2d(label == 1)<0)/nnz(label == 1);

FPost_3d = nnz(score_3d(label==0)<0)/n;
Accuracy_3d = nnz(score_3d(label == 1)<0)/nnz(label == 1);

if (verbose)
    display(['False positive 1d:      ' num2str(FPost_1d, '%5.3f') ]);
    display(['False positive 2d:      ' num2str(FPost_2d, '%5.3f') ]);
    display(['False positive 3d:      ' num2str(FPost_3d, '%5.3f') ]);
    
    display(['Accuracy 1d:      ' num2str(Accuracy_1d, '%5.3f') ]);
    display(['Accuracy 2d:      ' num2str(Accuracy_2d, '%5.3f') ]);
    display(['Accuracy 3d:      ' num2str(Accuracy_3d, '%5.3f') ]);
end