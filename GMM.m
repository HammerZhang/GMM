function [MODEL,c] = GMM( X,K_or_centroids )
% =====================================================================
% GMM is the exception-maximation interaction implentation of multivariate
% Gaussiam Mixture Model
%
% PX = GMM(X,K_or_centroids)
% [PX MODEL] = GMM(X,K_or_centroids)
%
% -X:N-by-D data matrix;
% -K_or_centroids: either K indicates the number of components or K-by-D
% matrix indicating the initial choosing of K controids
% 
%
% -PX: N-by-K matrix indicating the problity of each component generate
% each point in X
% -MODEL: a structure containing parameters of GMM:
%         MODEL.mu: a D-by-K matrix
%         MODEL.sigma: a D-by-D-by-K matrix 
%         MODEL.Pi: a 1-by-K vector
%
% -Author:Hammer Zhang
% -Time:2015-12-22
% =======================================================================
% 基本参数
termi_threshold = 1e-15;         % 迭代终止条件

% N样本数，D样本维数决定SGM是几维高斯分布
[N,D] = size(X);

% 指定混合高斯模型中独立高斯分布个数
if isscalar(K_or_centroids)
    K = K_or_centroids;
    rndp = randperm(N);
    centroids = X(rndp(1:K),:);
else 
    K = size(K_or_centroids,1);
    centroids = K_or_centroids;
end

% 设定初值
[pPi,pMu,pSigma] = InitParam();

pre_L = -inf;
pGamma = zeros(N,K);
PN = zeros(N,K);
inta_count = 0;
err_count = 0;
        
% use for plot error online
% figure;

% 迭代部分主体
while 1
    for i = 1 : K
        PN(:,i) = pPi(1,i) *  ComputePostProb(X,pMu(i,:),pSigma(:,:,i));
    end 
    Den = sum(PN,2);
    for j = 1 : K
        pGamma(:,j) = PN(:,j) ./ Den;
    end
    
    % 计算最大似然概率
    max_L = sum(log(Den));
    
    % 判断最大似然概率是否收敛
    t = max_L - pre_L;
    if t < termi_threshold 
        if err_count > 1000
            break;
        end
        err_count = err_count + 1;
    else
        err_count = 0;
    end
    
    % 迭代更新高斯分布系数
    pMuTemp = zeros(N,D);
    pSigmaTemp = zeros(D,D,N);
    Nk = sum(pGamma);
    for k = 1 : K
        for n = 1 : N
            pMuTemp(n,:) = pGamma(n,k) * X(n,:);           
        end
        pMu(k,:) = 1 / Nk(1,k) * sum(pMuTemp);
        for n = 1 : N
            pSigmaTemp(:,:,n) = pGamma(n,k) * (((X(n,:) - pMu(k,:))' * (X(n,:) - pMu(k,:))));
        end
        pSigma(:,:,k) = diag(diag(1 / Nk(1,k) * sum(pSigmaTemp,3)));
        pPi(1,k) = Nk(1,k) / N;
    end
    
    pre_L = max_L;
    inta_count = inta_count + 1;
    
    % print the progress
    if rem(inta_count,1000) == 0
        fprintf('computing the %f trial, the log likelihood is %f...\n',inta_count,max_L);
    end
end

% 设定初值函数
function [Pi,Mu,Sigma] = InitParam()
    Mu = centroids;
    Pi = zeros(1,K);
    Sigma = zeros(D,D,K);
    
    % hard assign x to each centroids
    distmat = repmat(sum(X.*X, 2), 1, K) + ...
        repmat(sum(Mu.*Mu, 2)', N, 1) - ...
        2*X*Mu';
    [~,labels] = min(distmat, [], 2);

    for ik=1:K
        Xk = X(labels == ik, :);
        Pi(ik) = size(Xk, 1)/N;
        Sigma(:, :, ik) = eye(D).*rand(D,D);
    end
end

% 计算后验概率函数
function PN = ComputePostProb(X,Mu,Sigma)
    %PN = mvnpdf(X,Mu,Sigma);
    PN = zeros(N,1);
    for cpn = 1 : N
        s_X = X(cpn,:) - Mu;
        e_P = -0.5 * (s_X * inv(Sigma) * s_X');
        den_para = 1 / sqrt((2 * pi)^D * det(Sigma));
        PN(cpn,1) = den_para * exp(e_P);
    end
end

% 输出分类结果
MODEL = [];
MODEL.Mu = pMu;
MODEL.Pi = pPi;
MODEL.Sigma = pSigma;
c = inta_count;

end

