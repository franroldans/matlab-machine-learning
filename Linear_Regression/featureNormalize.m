function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. 

for i=1:length(X(1,:))% Length of training set.
    mu(:,i) = mean(X(:,i));
    sigma(:,i) = std(X(:,i));
    X_norm(:,i)=(X(:,i)-mu(:,i))./sigma(:,i);   
end

end
