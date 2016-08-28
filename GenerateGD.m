function R = GenerateGD( Mu,Sigma,N)
% ========================================================================
% GENERATEGD generate a multivariate gaussian distribution, the number of
% single gaussian distribution is K.
%
% -Mu: a 1-by-D-by-K matrix contains the mean value of GMM
% -Sigma: a D-by-D-by-K matrix containing the covariance matrix
% -K: the number of single gaussian distribution
% -N: the number of each gaussian distribution sample
%
% -R: a K*N-by-D matrix contains the output of multivariate gaussian
% distribution
%
% ========================================================================

[a,D,K] = size(Mu);
R = zeros(N,D,K);
% 生成单独多元高斯分布
for k = 1 : K
    R(:,:,k) = mvnrnd(Mu(:,:,k),Sigma(:,:,k),N);
end

% 重新排列数组
R = reshape(R,N * K, D);



end

